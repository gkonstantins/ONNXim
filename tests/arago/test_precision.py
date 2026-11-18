"""Bit-exact tile and rounded/saturated output checks, on CPU or CUDA."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/arago"))
from check_attention import a16_wide_dot, native_dot, pack_int8, normalize
DEVICE=sys.argv.pop(1) if len(sys.argv)>1 else "cpu"
torch.backends.cuda.matmul.allow_tf32=False

class PrecisionTests(unittest.TestCase):
    def test_four_rank_functional_partition(self):
        # Independent dense MLP oracle for placement, local projections and sum.
        # Tensor columns partition the up projection; rows partition the down.
        torch.manual_seed(64)
        x=torch.randn(16,32,dtype=torch.float64)
        up=torch.randn(4,32,64,dtype=x.dtype)/8
        down=torch.randn(4,64,32,dtype=x.dtype)/8
        ref=x
        for a,b in zip(up,down):
            ref=ref+torch.relu(ref@a)@b
        for dp,tp,pp in ((4,1,1),(1,4,1),(1,1,4),(2,2,1),(2,1,2),(1,2,2)):
            parts=[]
            for h in x.chunk(dp,0):
                for stage in range(pp):
                    for layer in range(stage*(4//pp),(stage+1)*(4//pp)):
                        partials=[torch.relu(h@a)@b for a,b in
                                  zip(up[layer].chunk(tp,1),down[layer].chunk(tp,0))]
                        h=h+sum(partials)
                parts.append(h)
            torch.testing.assert_close(torch.cat(parts),ref)

    def test_rowblocked_causal_contract(self):
        # Naive and SRAM-row schedules may pick different output scales.
        # With exact INT8 products both compute the same quantized attention.
        torch.manual_seed(28)
        for batch in (1,16):
            g=torch.Generator().manual_seed(191)
            q=torch.randint(-11,12,(batch,4,9,128),generator=g).float().to(DEVICE)
            k=torch.randint(-11,12,(batch,1,17,128),generator=g).float().to(DEVICE)
            v=torch.randn(batch,1,17,128,generator=g).to(DEVICE)
            pos=torch.arange(8,17,device=DEVICE)
            keys=torch.arange(17,device=DEVICE)
            def evaluate(block):
                chunks=[]
                for start in range(0,9,block):
                    stop=min(start+block,9)
                    visible=8+stop
                    logits=(q[:,:,start:stop]@k[:,:,:visible].transpose(-1,-2))/128**.5
                    weights=normalize(logits,keys[:visible]<=pos[start:stop,None])
                    chunks.append(weights@v[:,:,:visible])
                return torch.cat(chunks,dim=2)
            torch.testing.assert_close(evaluate(4),evaluate(9),rtol=1e-4,atol=1e-5)

    def test_integer_contracts(self):
        rng=torch.Generator().manual_seed(51)
        for rows in (16,32):
            for k in (1,127,128,256):
                for bits,policy in ((8,"exact4"),(16,"a16exact")):
                    start=-(1<<(bits-1))
                    a=torch.randint(start,-start,(4,rows,k),generator=rng)
                    b=torch.randint(start,-start,(k,16),generator=rng)
                    a[:,0]=start
                    b[:,0]=start
                    wide_ref=a.long()@b.long()
                    if bits==16:
                        wide=a16_wide_dot(a.float().to(DEVICE),b.float().to(DEVICE))
                        self.assertTrue(torch.equal(wide.cpu().long(),wide_ref))
                    ref=wide_ref.float()
                    actual=native_dot(a.float().to(DEVICE),b.float().to(DEVICE),policy).cpu()
                    self.assertTrue(torch.equal(actual,ref))
        a=torch.full((4,16,128),127.,device=DEVICE)
        b=torch.full((128,16),127.,device=DEVICE)
        for policy in ("saturate","scaled"):
            out=native_dot(a,b,policy)
            self.assertEqual(out[0,0,0].item(),32767 if policy=="saturate" else 2064512)
        ties=torch.ones((4,16,128),device=DEVICE)
        for total,want in ((32,0),(96,128),(-32,0),(-96,-128)):
            other=torch.zeros((128,16),device=DEVICE)
            other[0]=total
            self.assertEqual(native_dot(ties,other,"fixedshift")[0,0,0].item(),want)
        x,_=pack_int8(a,-1,11)
        y,_=pack_int8(b,0,11)
        self.assertEqual(native_dot(x,y,"bounded")[0,0,0].item(),128*121)
        zero,scale=pack_int8(a*0,-1,11)
        self.assertFalse(zero.any())
        self.assertTrue((scale==1).all())

if __name__=="__main__":
    unittest.main()
