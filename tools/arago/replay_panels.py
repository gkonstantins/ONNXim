"""Replay a native streaming KV panel plan in PyTorch against dense causal attention."""
import argparse
from pathlib import Path
import tempfile
import torch
from architecture_study import DEFAULT_BINARY, simulate


def replay(row, q, k, v):
    # All users, one representative KV group. Each group has four query heads.
    past = row["visible_context"] - row["queries"]
    groups = {}
    for event in row["tile_plan"]:
        groups.setdefault(event["q0"], []).append(event)
    out = torch.zeros_like(q)
    position = 0
    for first,events in groups.items():
        count = events[0]["q"]
        if first != position:
            raise ValueError("missing or overlapping query panel")
        position += count
        query = q[:,:,first:first+count]
        peak = torch.full_like(query[..., :1],-torch.inf)
        mass, numerator = torch.zeros_like(peak), torch.zeros_like(query)
        releases = {}
        cursor = 0
        for e in events:
            slot=e["slot"]
            end=cursor+e["k"]
            if (e["k0"] != cursor or e["q"] != count or e["slot_ready"] < releases.get(slot,0)
                    or not e["slot_ready"] <= e.get("k_loaded",e["loaded"]) <= e["loaded"]
                    or not e.get("k_loaded",e["loaded"]) <= e.get("k_packed",e["loaded"]) <= e["score"]
                    or not e["loaded"] <= e.get("v_packed",e["loaded"]) <= e["pv"]
                    or not e["score"] <= e["pv"] <= e["released"]):
                raise ValueError("KV buffer reused early or invalid panel dependency")
            releases[slot] = e["released"]
            cursor=end
        # Replay actual load completions and consumers in virtual-clock order.
        # Two blocks may be resident at once. Tie order releases an old tile
        # before replacing it; reads require a completed DMA load.
        commands=[]
        for e in events:
            commands.extend([(e.get("k_loaded",e["loaded"]),1,"key",e),
                             (e["loaded"],1,"value",e), (e["score"],2,"score",e),
                             (e["pv"],0,"pv",e)])
        slots, pending = {}, {}
        for _,_,kind,e in sorted(commands,key=lambda cmd:cmd[:2]):
            start,end,slot=e["k0"],e["k0"]+e["k"],e["slot"]
            if kind=="key":
                slots[slot]=[start,k[:,:,start:end].clone(),None]
                continue
            if slot not in slots or slots[slot][0]!=start:
                raise ValueError("consumer observed the wrong tile in SRAM")
            _,keys,values=slots[slot]
            if kind=="value":
                slots[slot][2]=v[:,:,start:end].clone()
            elif kind=="score":
                mask = torch.arange(start,end,device=q.device) <= torch.arange(
                    past+first,past+first+count,device=q.device)[:,None]
                logits = (query @ keys.transpose(-1,-2) / 128**.5).masked_fill(~mask,-torch.inf)
                new_peak = torch.maximum(peak,logits.amax(-1,keepdim=True))
                factor = torch.exp(peak-new_peak)
                weights = torch.exp(logits-new_peak)
                mass = factor*mass + weights.sum(-1,keepdim=True)
                pending[start]=factor,weights
                peak = new_peak
            else:
                factor,weights=pending.pop(start)
                numerator = factor*numerator + weights @ values
        if cursor != past+first+count:
            raise ValueError("incorrect visible key range")
        out[:,:,first:first+count] = numerator/mass
    if position != row["queries"]:
        raise ValueError("incomplete query rows")
    return out


def verify(binary, device):
    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        for batch in (1,16):
            for double in (True,False):
                rows=simulate(binary,Path(tmp),batch,"flashattention",
                        {"trace_panels":True,"double_buffer":double,"key_tile":32,"optimized_decode":True},145,2)
                for row in rows:
                    nq,nk=row["queries"],row["visible_context"]
                    q=torch.randn(batch,4,nq,128,device=device)
                    k=torch.randn(batch,1,nk,128,device=device)
                    v=torch.randn_like(k)
                    mask=torch.arange(nk,device=device)<=torch.arange(nk-nq,nk,device=device)[:,None]
                    ref=((q@k.transpose(-1,-2)/128**.5).masked_fill(~mask,-torch.inf)).softmax(-1)@v
                    torch.testing.assert_close(replay(row,q,k,v),ref,atol=3e-6,rtol=2e-5)
                # Deliberately corrupt a real plan. A numerical pass alone can miss this hazard.
                last = rows[-1]["tile_plan"][-1]
                for field,bad in (("slot_ready",-1),("k_packed",last["score"]+1),
                                  ("v_packed",last["pv"]+1)):
                    saved = last[field]
                    last[field] = bad
                    try:
                        replay(rows[-1],q,k,v)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError(f"accepted invalid {field}")
                    last[field] = saved



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulator",type=Path,default=DEFAULT_BINARY)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=p.parse_args()
    verify(args.simulator.resolve(),args.device)
    print("Panel replays and injected hazard passed")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulator",type=Path,default=DEFAULT_BINARY)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=p.parse_args()
    verify(args.simulator.resolve(),args.device)
    print("Panel replays and injected hazard passed")
