# First token, sustained generation and kernel bottleneck by case

The simulated workload gives every user a 2,048-token prompt and generates 256 output tokens. With 16 simultaneous users, naive attention produces the first token at 686.92 s (11 min 27 s). FlashAttention produces it at 670.25 s (11 min 10 s). Once prefill ends, each user receives one token every 1.526 s on average with FlashAttention. The group receives 16 tokens per step, or 10.48 tokens/s. Mean time between tokens (TBT) covers the 255 cached steps. Token 256 arrives at 1,059.44 s (17 min 39 s) under this schedule.

The earlier 309.25 s prefill figure divides about 237.51 trillion useful MACs by the ideal 768 GMAC/s. The scheduled model accounts for padded or underfilled tiles, data movement, command delivery and output drain. It also includes embedding and final-token endpoints.

One device spends 34.73 s of the 670.25 s prefill in attention across 32 layers. The remaining 635.52 s cover projections, the feed-forward network, other block operations and endpoints. In decode, attention takes 0.411 s of each 1.526 s step. A kernel optimization therefore has much more scope to change decode than prefill in this mapping.

Device TTFT starts with weights resident and all prompts admitted together. Four-device TTFT additionally includes the configured host transfers/service and covers the last user’s first token. Wall-clock loading, tokenization and streaming to the client are outside these times. Precision choices in the timing model are assumptions; layer-0 numerical accuracy is available in the comparison.

[Four-device parallelism, ceilings and 70% bandwidth by case](four-case-metrics.md) · [Interactivity recommendations](interactivity-recommendations.md) · [Accuracy and utilization](architecture-comparison.md) · [Machine-readable case table](../example/arago/results/case-metrics.json) · [Paired perturbations and signed elasticities](../example/arago/results/case-elasticity.json)

## Reading the bottleneck columns

Each row has its own kernel sensitivity experiment. The PDF columns give the largest absolute time elasticity among the swept hardware parameters specified by Arago. The extended columns also include the assumed per-cell MAC throughput. Each phase has its own winner. An elasticity of 0.30 means a 10% parameter change in the tested direction changes kernel time by about 3% near that operating point. Winners apply to the selected attention algorithm and, for four devices, to the local shard. Whole-model time and host-link sensitivity have different denominators. The raw sweep file records each parameter, perturbation, and cycle count.

An array command latency win means the control CPU issues enough small tiles for command delivery to remain on the critical path. A SIMD command latency win points to frequently issued vector or reduction passes. A port-width win reflects bytes on that port. A MAC throughput win reflects time in matrix service. A zero local slope can arise from another bottleneck, integer cycle rounding, or an allocation that already fits.

The saturation and bounded-input modes failed the layer-0 accuracy checks for both batch sizes. Their timing rows show how quickly a numerically unacceptable result could be generated. The full-checkpoint row-blocked schedule still needs an accuracy run.

## Batch 1 design variants

| Case | TTFT s | TBT ms | Tokens/s/user | Tokens/s, all | Token 256 at s | PDF prefill | PDF decode | Extended prefill | Extended decode |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| baseline | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| naive | 43.27 | 1123.21 | 0.8903 | 0.890 | 329.69 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| exact4 (`precision_policy="exact4"`) | 151.60 | 3854.61 | 0.2594 | 0.259 | 1134.53 | array command latency 0.302 | array command latency 0.290 | MAC throughput 0.327 | MAC throughput 0.314 |
| bounded (`precision_policy="bounded"`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| saturate (`precision_policy="saturate"`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| queued_delivery (`queued_delivery=true`) | 34.43 | 895.44 | 1.1168 | 1.117 | 262.77 | SIMD port width 0.282 | SIMD command latency 0.219 | SIMD port width 0.282 | MAC throughput 0.405 |
| duplex_dma (`duplex_dma=true`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| transpose_decode (`transpose_decode=true`) | 42.22 | 985.73 | 1.0145 | 1.014 | 293.58 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| overlap_drain (`overlap_drain=false`) | 52.12 | 1215.04 | 0.8230 | 0.823 | 361.95 | array output width 0.246 | array command latency 0.239 | array output width 0.246 | MAC throughput 0.259 |
| double_buffer (`double_buffer=false`) | 42.45 | 1116.06 | 0.8960 | 0.896 | 327.04 | array command latency 0.271 | array command latency 0.213 | MAC throughput 0.294 | MAC throughput 0.231 |
| wavefront_fill (`wavefront_fill=true`) | 45.90 | 1165.57 | 0.8579 | 0.858 | 343.12 | array command latency 0.265 | array command latency 0.255 | MAC throughput 0.284 | MAC throughput 0.276 |
| query_tile (`query_tile=16`) | 42.98 | 1108.49 | 0.9021 | 0.902 | 325.64 | array command latency 0.297 | array command latency 0.276 | MAC throughput 0.318 | MAC throughput 0.299 |
| key_tile (`key_tile=64`) | 42.86 | 1117.01 | 0.8952 | 0.895 | 327.69 | array command latency 0.342 | array command latency 0.309 | array command latency 0.342 | array command latency 0.309 |
| double_output_port (`array_output_bytes_per_cycle=16`) | 42.06 | 1100.87 | 0.9084 | 0.908 | 322.78 | array command latency 0.304 | array command latency 0.279 | MAC throughput 0.326 | MAC throughput 0.302 |
| double_both_dma_links (`dma_bytes_per_cycle=128, sram_dma_rate=128`) | 40.32 | 1048.72 | 0.9535 | 0.954 | 307.74 | array command latency 0.301 | array command latency 0.277 | MAC throughput 0.326 | MAC throughput 0.300 |
| sweep_dma_bytes_per_cycle (`dma_bytes_per_cycle=70, sram_dma_rate=64`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_sram_dma_rate (`sram_dma_rate=70`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array16_input_bytes_per_cycle (`array16_input_bytes_per_cycle=70`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array32_input_bytes_per_cycle (`array32_input_bytes_per_cycle=106`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array_output_bytes_per_cycle (`array_output_bytes_per_cycle=9`) | 42.19 | 1106.82 | 0.9035 | 0.903 | 324.43 | array command latency 0.302 | array command latency 0.276 | MAC throughput 0.324 | MAC throughput 0.300 |
| sweep_vector_bytes_per_cycle (`vector_bytes_per_cycle=141`) | 41.53 | 1104.88 | 0.9051 | 0.905 | 323.27 | array command latency 0.305 | array command latency 0.280 | MAC throughput 0.330 | MAC throughput 0.303 |
| sweep_dma_setup (`dma_setup=180`) | 42.17 | 1099.05 | 0.9099 | 0.910 | 322.43 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_dram_read_start (`dram_read_start=180`) | 42.17 | 1099.30 | 0.9097 | 0.910 | 322.49 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_dram_write_start (`dram_write_start=135`) | 42.22 | 1108.30 | 0.9023 | 0.902 | 324.84 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array_launch (`array_launch=90`) | 41.43 | 1089.46 | 0.9179 | 0.918 | 319.24 | array command latency 0.275 | array command latency 0.255 | MAC throughput 0.332 | MAC throughput 0.307 |
| sweep_vector_launch (`vector_launch=270`) | 42.07 | 1100.85 | 0.9084 | 0.908 | 322.78 | array command latency 0.300 | array command latency 0.280 | MAC throughput 0.325 | MAC throughput 0.304 |
| sweep_scratchpad_bytes (`scratchpad_bytes=1153434`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| rowblocked | 48.88 | 1117.73 | 0.8947 | 0.895 | 333.90 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| independent_heads (`share_kv=false`) | 43.65 | 1184.92 | 0.8439 | 0.844 | 345.80 | array command latency 0.241 | array command latency 0.277 | MAC throughput 0.260 | MAC throughput 0.300 |
| sweep_mac_rate (`mac_percent=110`) | 40.47 | 1065.58 | 0.9385 | 0.938 | 312.19 | array command latency 0.308 | array command latency 0.284 | MAC throughput 0.334 | MAC throughput 0.308 |
| a16exact (`precision_policy="a16exact"`) | 334.83 | 7513.68 | 0.1331 | 0.133 | 2250.82 | array command latency 0.285 | array command latency 0.308 | MAC throughput 0.309 | MAC throughput 0.334 |
| fixedshift (`precision_policy="fixedshift"`) | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |

## Batch 16 design variants

| Case | TTFT s | TBT ms | Tokens/s/user | Tokens/s, all | Token 256 at s | PDF prefill | PDF decode | Extended prefill | Extended decode |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| baseline | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| naive | 686.92 | 1761.84 | 0.5676 | 9.081 | 1136.19 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| exact4 (`precision_policy="exact4"`) | 2405.51 | 5352.99 | 0.1868 | 2.989 | 3770.52 | array command latency 0.302 | array command latency 0.290 | MAC throughput 0.327 | MAC throughput 0.314 |
| bounded (`precision_policy="bounded"`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| saturate (`precision_policy="saturate"`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| queued_delivery (`queued_delivery=true`) | 546.49 | 1209.83 | 0.8266 | 13.225 | 855.00 | SIMD port width 0.282 | SIMD command latency 0.219 | SIMD port width 0.282 | MAC throughput 0.405 |
| duplex_dma (`duplex_dma=true`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| transpose_decode (`transpose_decode=true`) | 670.24 | 1417.20 | 0.7056 | 11.290 | 1031.63 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| overlap_drain (`overlap_drain=false`) | 827.38 | 1692.12 | 0.5910 | 9.456 | 1258.87 | array output width 0.246 | array command latency 0.239 | array output width 0.246 | MAC throughput 0.259 |
| double_buffer (`double_buffer=false`) | 673.74 | 1647.39 | 0.6070 | 9.712 | 1093.83 | array command latency 0.271 | array command latency 0.213 | MAC throughput 0.294 | MAC throughput 0.231 |
| wavefront_fill (`wavefront_fill=true`) | 728.60 | 1615.16 | 0.6191 | 9.906 | 1140.46 | array command latency 0.265 | array command latency 0.255 | MAC throughput 0.284 | MAC throughput 0.276 |
| query_tile (`query_tile=16`) | 682.25 | 1526.22 | 0.6552 | 10.483 | 1071.44 | array command latency 0.297 | array command latency 0.276 | MAC throughput 0.318 | MAC throughput 0.299 |
| key_tile (`key_tile=64`) | 680.30 | 1662.59 | 0.6015 | 9.624 | 1104.26 | array command latency 0.342 | array command latency 0.309 | array command latency 0.342 | array command latency 0.309 |
| double_output_port (`array_output_bytes_per_cycle=16`) | 667.63 | 1514.30 | 0.6604 | 10.566 | 1053.78 | array command latency 0.304 | array command latency 0.279 | MAC throughput 0.326 | MAC throughput 0.302 |
| double_both_dma_links (`dma_bytes_per_cycle=128, sram_dma_rate=128`) | 639.71 | 1449.55 | 0.6899 | 11.038 | 1009.34 | array command latency 0.301 | array command latency 0.277 | MAC throughput 0.326 | MAC throughput 0.300 |
| sweep_dma_bytes_per_cycle (`dma_bytes_per_cycle=70, sram_dma_rate=64`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_sram_dma_rate (`sram_dma_rate=70`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array16_input_bytes_per_cycle (`array16_input_bytes_per_cycle=70`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array32_input_bytes_per_cycle (`array32_input_bytes_per_cycle=106`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array_output_bytes_per_cycle (`array_output_bytes_per_cycle=9`) | 669.68 | 1523.61 | 0.6563 | 10.501 | 1058.20 | array command latency 0.302 | array command latency 0.276 | MAC throughput 0.324 | MAC throughput 0.300 |
| sweep_vector_bytes_per_cycle (`vector_bytes_per_cycle=141`) | 659.03 | 1516.67 | 0.6593 | 10.549 | 1045.78 | array command latency 0.305 | array command latency 0.280 | MAC throughput 0.330 | MAC throughput 0.303 |
| sweep_dma_setup (`dma_setup=180`) | 669.29 | 1516.41 | 0.6595 | 10.551 | 1055.97 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_dram_read_start (`dram_read_start=180`) | 669.39 | 1516.75 | 0.6593 | 10.549 | 1056.17 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_dram_write_start (`dram_write_start=135`) | 670.17 | 1525.96 | 0.6553 | 10.485 | 1059.29 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| sweep_array_launch (`array_launch=90`) | 657.57 | 1496.57 | 0.6682 | 10.691 | 1039.20 | array command latency 0.275 | array command latency 0.255 | MAC throughput 0.332 | MAC throughput 0.307 |
| sweep_vector_launch (`vector_launch=270`) | 667.73 | 1512.38 | 0.6612 | 10.579 | 1053.38 | array command latency 0.300 | array command latency 0.280 | MAC throughput 0.325 | MAC throughput 0.304 |
| sweep_scratchpad_bytes (`scratchpad_bytes=1153434`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| rowblocked | 776.68 | 1674.16 | 0.5973 | 9.557 | 1203.59 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| independent_heads (`share_kv=false`) | 692.97 | 2749.24 | 0.3637 | 5.820 | 1394.03 | array command latency 0.241 | array command latency 0.277 | MAC throughput 0.260 | MAC throughput 0.300 |
| sweep_mac_rate (`mac_percent=110`) | 642.36 | 1471.81 | 0.6794 | 10.871 | 1017.67 | array command latency 0.308 | array command latency 0.284 | MAC throughput 0.334 | MAC throughput 0.308 |
| a16exact (`precision_policy="a16exact"`) | 5319.33 | 13054.69 | 0.0766 | 1.226 | 8648.27 | array command latency 0.285 | array command latency 0.308 | MAC throughput 0.309 | MAC throughput 0.334 |
| fixedshift (`precision_policy="fixedshift"`) | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |

## Batch, attention schedule and precision

| Case | TTFT s | TBT ms | Tokens/s/user | Tokens/s, all | Token 256 at s | PDF prefill | PDF decode | Extended prefill | Extended decode |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| 1/naive/scaled | 43.27 | 1123.21 | 0.8903 | 0.890 | 329.69 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 1/naive/saturate | 43.27 | 1123.21 | 0.8903 | 0.890 | 329.69 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 1/naive/bounded | 43.27 | 1123.21 | 0.8903 | 0.890 | 329.69 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 1/naive/exact4 | 152.64 | 3869.34 | 0.2584 | 0.258 | 1139.33 | array command latency 0.270 | array command latency 0.252 | MAC throughput 0.292 | MAC throughput 0.273 |
| 1/naive/a16exact | 335.87 | 7528.40 | 0.1328 | 0.133 | 2255.61 | array command latency 0.277 | array command latency 0.296 | MAC throughput 0.300 | MAC throughput 0.321 |
| 1/naive/fixedshift | 43.27 | 1123.21 | 0.8903 | 0.890 | 329.69 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 1/rowblocked/scaled | 48.88 | 1117.73 | 0.8947 | 0.895 | 333.90 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 1/rowblocked/saturate | 48.88 | 1117.73 | 0.8947 | 0.895 | 333.90 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 1/rowblocked/bounded | 48.88 | 1117.73 | 0.8947 | 0.895 | 333.90 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 1/rowblocked/exact4 | 169.47 | 3863.86 | 0.2588 | 0.259 | 1154.75 | array command latency 0.263 | array command latency 0.265 | MAC throughput 0.280 | MAC throughput 0.287 |
| 1/rowblocked/a16exact | 390.75 | 7522.92 | 0.1329 | 0.133 | 2309.09 | array command latency 0.301 | array command latency 0.301 | MAC throughput 0.321 | MAC throughput 0.326 |
| 1/rowblocked/fixedshift | 48.88 | 1117.73 | 0.8947 | 0.895 | 333.90 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 1/flashattention/scaled | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/flashattention/saturate | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/flashattention/bounded | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/flashattention/exact4 | 151.60 | 3854.61 | 0.2594 | 0.259 | 1134.53 | array command latency 0.302 | array command latency 0.290 | MAC throughput 0.327 | MAC throughput 0.314 |
| 1/flashattention/a16exact | 334.83 | 7513.68 | 0.1331 | 0.133 | 2250.82 | array command latency 0.285 | array command latency 0.308 | MAC throughput 0.309 | MAC throughput 0.334 |
| 1/flashattention/fixedshift | 42.23 | 1108.49 | 0.9021 | 0.902 | 324.89 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 16/naive/scaled | 686.92 | 1761.84 | 0.5676 | 9.081 | 1136.19 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 16/naive/saturate | 686.92 | 1761.84 | 0.5676 | 9.081 | 1136.19 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 16/naive/bounded | 686.92 | 1761.84 | 0.5676 | 9.081 | 1136.19 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 16/naive/exact4 | 2422.18 | 5588.62 | 0.1789 | 2.863 | 3847.27 | array command latency 0.270 | array command latency 0.252 | MAC throughput 0.292 | MAC throughput 0.273 |
| 16/naive/a16exact | 5335.99 | 13290.31 | 0.0752 | 1.204 | 8725.02 | array command latency 0.277 | array command latency 0.296 | MAC throughput 0.300 | MAC throughput 0.321 |
| 16/naive/fixedshift | 686.92 | 1761.84 | 0.5676 | 9.081 | 1136.19 | array command latency 0.202 | array command latency 0.175 | MAC throughput 0.218 | MAC throughput 0.190 |
| 16/rowblocked/scaled | 776.68 | 1674.16 | 0.5973 | 9.557 | 1203.59 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 16/rowblocked/saturate | 776.68 | 1674.16 | 0.5973 | 9.557 | 1203.59 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 16/rowblocked/bounded | 776.68 | 1674.16 | 0.5973 | 9.557 | 1203.59 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 16/rowblocked/exact4 | 2691.40 | 5500.93 | 0.1818 | 2.909 | 4094.13 | array command latency 0.263 | array command latency 0.265 | MAC throughput 0.280 | MAC throughput 0.287 |
| 16/rowblocked/a16exact | 6214.07 | 13202.63 | 0.0757 | 1.212 | 9580.74 | array command latency 0.301 | array command latency 0.301 | MAC throughput 0.321 | MAC throughput 0.326 |
| 16/rowblocked/fixedshift | 776.68 | 1674.16 | 0.5973 | 9.557 | 1203.59 | array command latency 0.197 | array command latency 0.203 | MAC throughput 0.210 | MAC throughput 0.220 |
| 16/flashattention/scaled | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 16/flashattention/saturate | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 16/flashattention/bounded | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 16/flashattention/exact4 | 2405.51 | 5352.99 | 0.1868 | 2.989 | 3770.52 | array command latency 0.302 | array command latency 0.290 | MAC throughput 0.327 | MAC throughput 0.314 |
| 16/flashattention/a16exact | 5319.33 | 13054.69 | 0.0766 | 1.226 | 8648.27 | array command latency 0.285 | array command latency 0.308 | MAC throughput 0.309 | MAC throughput 0.334 |
| 16/flashattention/fixedshift | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |

## Four accelerators on one host

Case names give DP / TP / PP / microbatches / host model. Each local kernel ranking is reused across host models that run the same shard. [Transport assumptions and per-case bandwidth](four-case-metrics.md).

| Case | TTFT s | TBT ms | Tokens/s/user | Tokens/s, all | Token 256 at s | PDF prefill | PDF decode | Extended prefill | Extended decode |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| 4/1/1/1/serial | 167.83 | 1191.91 | 0.8390 | 13.424 | 471.77 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 4/1/1/1/chunked | 167.83 | 1191.91 | 0.8390 | 13.424 | 471.77 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 4/1/1/1/overlap_bound | 167.83 | 1191.91 | 0.8390 | 13.424 | 471.77 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 4/1/1/1/serial_1us | 167.83 | 1191.91 | 0.8390 | 13.424 | 471.77 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 4/1/1/1/finite_host | 167.83 | 1191.92 | 0.8390 | 13.424 | 471.77 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/4/1/1/serial | 179.62 | 446.96 | 2.2373 | 35.797 | 293.60 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/4/1/1/chunked | 175.33 | 444.87 | 2.2479 | 35.966 | 288.77 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/4/1/1/overlap_bound | 171.03 | 442.77 | 2.2585 | 36.136 | 283.94 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/4/1/1/serial_1us | 179.62 | 447.09 | 2.2367 | 35.787 | 293.63 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/4/1/1/finite_host | 181.34 | 447.94 | 2.2324 | 35.719 | 295.57 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/1/serial | 670.35 | 1526.27 | 0.6552 | 10.483 | 1059.55 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/1/chunked | 670.30 | 1526.24 | 0.6552 | 10.483 | 1059.49 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/1/overlap_bound | 670.25 | 1526.22 | 0.6552 | 10.483 | 1059.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/1/serial_1us | 670.35 | 1526.28 | 0.6552 | 10.483 | 1059.55 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/1/finite_host | 670.35 | 1526.29 | 0.6552 | 10.483 | 1059.55 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/2/serial | 419.68 | 1686.87 | 0.5928 | 9.485 | 849.83 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/2/chunked | 419.63 | 1686.84 | 0.5928 | 9.485 | 849.78 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/2/overlap_bound | 419.58 | 1686.82 | 0.5928 | 9.485 | 849.72 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/2/serial_1us | 419.68 | 1686.88 | 0.5928 | 9.485 | 849.84 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/2/finite_host | 419.68 | 1686.89 | 0.5928 | 9.485 | 849.84 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/4/serial | 293.97 | 2257.63 | 0.4429 | 7.087 | 869.67 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/4/chunked | 293.92 | 2257.61 | 0.4429 | 7.087 | 869.61 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/4/overlap_bound | 293.87 | 2257.58 | 0.4430 | 7.087 | 869.55 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/4/serial_1us | 293.97 | 2257.66 | 0.4429 | 7.087 | 869.68 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/1/4/4/finite_host | 293.97 | 2257.67 | 0.4429 | 7.087 | 869.68 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/2/1/1/serial | 173.28 | 693.46 | 1.4420 | 23.073 | 350.12 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/2/1/1/chunked | 171.14 | 692.41 | 1.4442 | 23.108 | 347.70 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/2/1/1/overlap_bound | 168.99 | 691.37 | 1.4464 | 23.143 | 345.29 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/2/1/1/serial_1us | 173.28 | 693.59 | 1.4418 | 23.068 | 350.15 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/2/1/1/finite_host | 174.31 | 694.11 | 1.4407 | 23.051 | 351.31 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/1/serial | 335.66 | 1303.32 | 0.7673 | 12.276 | 668.00 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/1/chunked | 335.64 | 1303.31 | 0.7673 | 12.276 | 667.99 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/1/overlap_bound | 335.62 | 1303.30 | 0.7673 | 12.276 | 667.97 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/1/serial_1us | 335.66 | 1303.32 | 0.7673 | 12.276 | 668.01 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/1/finite_host | 335.66 | 1303.33 | 0.7673 | 12.276 | 668.01 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/2/serial | 251.82 | 1826.04 | 0.5476 | 8.762 | 717.46 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/2/chunked | 251.80 | 1826.03 | 0.5476 | 8.762 | 717.44 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/2/overlap_bound | 251.78 | 1826.02 | 0.5476 | 8.762 | 717.42 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/2/serial_1us | 251.82 | 1826.05 | 0.5476 | 8.762 | 717.46 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/2/finite_host | 251.82 | 1826.06 | 0.5476 | 8.762 | 717.46 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/4/serial | 211.18 | 2954.74 | 0.3384 | 5.415 | 964.64 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/4/chunked | 211.16 | 2954.74 | 0.3384 | 5.415 | 964.62 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/4/overlap_bound | 211.15 | 2954.73 | 0.3384 | 5.415 | 964.60 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/4/serial_1us | 211.18 | 2954.76 | 0.3384 | 5.415 | 964.64 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 2/1/2/4/finite_host | 211.18 | 2954.77 | 0.3384 | 5.415 | 964.65 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/1/serial | 341.78 | 806.00 | 1.2407 | 19.851 | 547.31 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/1/chunked | 339.62 | 804.94 | 1.2423 | 19.877 | 544.88 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/1/overlap_bound | 337.44 | 803.88 | 1.2440 | 19.903 | 542.43 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/1/serial_1us | 341.78 | 806.13 | 1.2405 | 19.848 | 547.35 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/1/finite_host | 342.81 | 806.65 | 1.2397 | 19.835 | 548.51 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/2/serial | 257.85 | 1077.63 | 0.9280 | 14.847 | 532.65 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/2/chunked | 255.69 | 1076.57 | 0.9289 | 14.862 | 530.22 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/2/overlap_bound | 253.51 | 1075.50 | 0.9298 | 14.877 | 527.76 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/2/serial_1us | 257.85 | 1077.89 | 0.9277 | 14.844 | 532.72 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/2/finite_host | 258.89 | 1078.40 | 0.9273 | 14.837 | 533.88 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/4/serial | 215.75 | 1704.30 | 0.5868 | 9.388 | 650.35 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/4/chunked | 213.59 | 1703.24 | 0.5871 | 9.394 | 647.91 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/4/overlap_bound | 211.41 | 1702.17 | 0.5875 | 9.400 | 645.46 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/4/serial_1us | 215.75 | 1704.82 | 0.5866 | 9.385 | 650.48 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |
| 1/2/2/4/finite_host | 216.78 | 1705.34 | 0.5864 | 9.382 | 651.64 | array command latency 0.299 | array command latency 0.276 | MAC throughput 0.323 | MAC throughput 0.299 |

## Reproduce

Build the focused ONNXim target and run the commands in the [coverage report](coverage.md#reproduce). The original case manifest defines 64 single-device rows, 36 crossed workload rows and 60 four-device rows. Several comparisons share the same baseline. Hardware unit tests for a descriptor, tile, buffer hazard or integer dot have their own cycle, dependency or numerical oracle. TTFT requires an entire model and request trace.
