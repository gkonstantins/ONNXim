"""Integration contracts and independent time/byte oracles for architecture sweeps."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/arago"))
import architecture_study as study

BINARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 else study.DEFAULT_BINARY
class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory=Path(tmp.name)

    def run_case(self,b=1,mode="flashattention",**cfg):
        return study.simulate(BINARY,self.directory,b,mode,cfg,160,4)

    def test_assignment_constants(self):
        c=json.loads(study.BASE.read_text())
        self.assertEqual([c[k] for k in ("dram_bytes","scratchpad_bytes","clock_hz")],[64<<30,1<<20,10**9])
        self.assertEqual([c[k] for k in ("dma_setup","dram_read_start","dram_write_start","dma_bytes_per_cycle",
            "array_launch","vector_launch","array16_input_bytes_per_cycle","array32_input_bytes_per_cycle",
            "array_output_bytes_per_cycle","vector_bytes_per_cycle")],[200,200,150,64,100,300,64,96,8,128])

    def test_factorial_batch_schedule_precision(self):
        for batch in (1,16):
            for mode in ("naive","rowblocked","flashattention"):
                one=self.run_case(batch,mode)
                for name in ("scaled","fixedshift","saturate","bounded","exact4","a16exact"):
                    with self.subTest(batch=batch,mode=mode,precision=name):
                        rows=self.run_case(batch,mode,precision_policy=name)
                        for a,b in zip(one,rows):
                            for field in ("array16_tiles","array32_tiles"):
                                factor={"exact4":4,"a16exact":16}.get(name,1)
                                self.assertEqual(b["attention"][field],a["attention"][field]*factor)
                            if name in ("exact4","a16exact"):
                                self.assertGreater(b["layer"]["cycles"],a["layer"]["cycles"])
                            else:
                                self.assertEqual(b["layer"],a["layer"])
                            self.assertGreater(b["endpoints"]["cycles"],0)

    def test_placements(self):
        for tp in (1,2,4):
            base=self.run_case(16)
            sharded=self.run_case(16,tensor_shards=tp)
            self.assertEqual(sharded[0]["attention"]["cycles"]*tp,base[0]["attention"]["cycles"])
            self.assertEqual(sharded[0]["endpoints"],base[0]["endpoints"])
            self.assertLessEqual(sharded[0]["layer"]["cycles"],base[0]["layer"]["cycles"])
            if tp>1:
                self.assertLess(sharded[0]["resident_estimate_B"],base[0]["resident_estimate_B"])
        staged=self.run_case(16,pipeline_stages=4)
        self.assertLess(staged[0]["resident_estimate_B"],base[0]["resident_estimate_B"])
        self.assertEqual(staged[0]["layer"],base[0]["layer"])

    def test_mapping_choices(self):
        base=self.run_case(16)
        for key,opt in (("queued_delivery",True),("duplex_dma",True),("overlap_drain",False),
                        ("double_buffer",False),("wavefront_fill",True),("transpose_decode",True)):
            other=self.run_case(16,**{key:opt})
            for a,b in zip(base,other):
                self.assertEqual(a["layer"]["dram_read_bytes"],b["layer"]["dram_read_bytes"])
                self.assertEqual(a["layer"]["dram_write_bytes"],b["layer"]["dram_write_bytes"])
            if key in ("overlap_drain","wavefront_fill","double_buffer"):
                self.assertGreater(other[0]["attention"]["cycles"],base[0]["attention"]["cycles"])
            if key=="queued_delivery":
                self.assertLess(other[0]["layer"]["cycles"],base[0]["layer"]["cycles"])
            if key=="transpose_decode":
                self.assertGreater(other[1]["layer"]["array32_tiles"],0)
        self.assertGreater(self.run_case(16,share_kv=False)[0]["attention"]["dram_read_bytes"],base[0]["attention"]["dram_read_bytes"])
        for key,val in (("key_tile",64),("query_tile",16)):
            other=self.run_case(16,**{key:val})
            self.assertLess(other[0]["attention"]["peak_sram_bytes"],base[0]["attention"]["peak_sram_bytes"])
        for overrides in ({"sram_dma_rate":0},{"scratchpad_bytes":16},{"dram_bytes":1},
                          {"precision_policy":"guess"},{"tensor_shards":3},{"key_tile":257}):
            with self.subTest(overrides=overrides),self.assertRaises(study.subprocess.CalledProcessError):
                self.run_case(**overrides)

    def test_sweep_monotonic_and_payload(self):
        base=self.run_case(16)
        for case in json.loads(study.DESIGN.read_text())["cases"]:
            if "parameter_ratio" not in case:
                continue
            with self.subTest(case=case["name"]):
                variant=self.run_case(16,**case["override"])
                for a,b in zip(base,variant):
                    self.assertLessEqual(b["layer"]["cycles"],a["layer"]["cycles"])
                    self.assertEqual(b["layer"]["dram_read_bytes"],a["layer"]["dram_read_bytes"])

    def test_other_microbatches_keep_their_kv(self):
        row = {"batch":4, "visible_context":2303, "resident_estimate_B":700}
        self.assertEqual(study.rank_resident_bytes(row,2,2,1),700)
        # Three paused groups of four users, sixteen layers, 512 K/V channels, A16.
        self.assertEqual(study.rank_resident_bytes(row,2,2,4),700+12*2303*16*512*4)

    def test_utilization_uses_elapsed_cycles(self):
        rows=self.run_case(16,"flashattention")[1:]
        counters=study.attention_counters(rows)
        for key,ratio in counters["utilization"].items():
            self.assertGreaterEqual(ratio,0)
            self.assertLessEqual(ratio,1)
            self.assertAlmostEqual(ratio,sum(r["attention"][key] for r in rows) /
                                   sum(r["attention"]["cycles"] for r in rows))

    def test_pipeline_stage_recurrence(self):
        row = {"layer":{"cycles":10},"endpoints":{"cycles":12},"embedding_cycles":2}
        self.assertEqual(study.pipeline_cycles(row,4,1),332)
        self.assertEqual(study.pipeline_cycles(row,4,4),332+3*90)
        self.assertEqual(study.pipeline_cycles(row,1,4),4*332)
        self.assertAlmostEqual(study.ideal_roof()["tokens_s_user"],21.21,delta=.02)

    def test_host_cut_and_threshold(self):
        self.assertEqual(study.host_payload(16,1,4,1,1),(64,64,2))
        self.assertEqual(study.host_payload(16,1,1,4,1),(33554496,33554688,130))
        self.assertEqual(study.host_payload(16,1,1,1,4),(393280,393280,8))
        for dp,tp,pp in json.loads(study.DESIGN.read_text())["placements"]:
            payload=study.host_payload(16,1,dp,tp,pp)
            for options in json.loads(study.DESIGN.read_text())["host_models"].values():
                compute=4e8
                bw=study.bandwidth_for_fraction(compute,*payload,0.7,options)
                ratio=lambda x: compute/study.network_time(compute,*payload,x,options)
                self.assertAlmostEqual(ratio(bw),0.7)
                self.assertLess(ratio(bw*.99),.7)
        self.assertEqual(study.host_service(16,1,4,{"reduce_GB_s":64}),655360)
        self.assertEqual(study.host_service(16,1,1,{"host_dispatch_ns":15}),15)
        impossible={"startup_ns":10**8,"duplex":False,"overlap":True}
        self.assertIsNone(study.bandwidth_for_fraction(100,64,64,2,.7,impossible))

if __name__=="__main__":
    unittest.main()
