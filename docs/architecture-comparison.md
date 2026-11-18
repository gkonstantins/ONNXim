# Architecture comparison

Each device result uses 2,048 prompt tokens and 256 outputs per user. TTFT includes embedding, 32 decoder blocks, final norm, vocabulary head and greedy selection on a warm device. Decode uses the mean over all 255 cached positions. A synchronized batch emits one token per user per step. [Assumptions and requirements](coverage.md) · [Raw results](../example/arago/results/architecture-study.json)

## Batch 1, one variable at a time

| Case | Device TTFT (s) | Decode (ms/step) | Tokens/s/user | Speedup vs baseline | Prefill / decode attention elasticity |
|---|---:|---:|---:|---:|---:|
| baseline | 42.228 | 1108.485 | 0.9021 | 1.0000× | — |
| naive | 43.270 | 1123.212 | 0.8903 | 0.9869× | — |
| exact4 | 151.603 | 3854.613 | 0.2594 | 0.2876× | — |
| bounded | 42.228 | 1108.485 | 0.9021 | 1.0000× | — |
| saturate | 42.228 | 1108.485 | 0.9021 | 1.0000× | — |
| queued_delivery | 34.434 | 895.439 | 1.1168 | 1.2379× | — |
| duplex_dma | 42.228 | 1108.485 | 0.9021 | 1.0000× | — |
| transpose_decode | 42.220 | 985.729 | 1.0145 | 1.1245× | — |
| overlap_drain | 52.115 | 1215.044 | 0.8230 | 0.9123× | — |
| double_buffer | 42.447 | 1116.059 | 0.8960 | 0.9932× | — |
| wavefront_fill | 45.903 | 1165.574 | 0.8579 | 0.9510× | — |
| query_tile | 42.978 | 1108.485 | 0.9021 | 1.0000× | — |
| key_tile | 42.856 | 1117.009 | 0.8952 | 0.9924× | — |
| double_output_port | 42.060 | 1100.870 | 0.9084 | 1.0069× | — |
| double_both_dma_links | 40.316 | 1048.725 | 0.9535 | 1.0570× | — |
| sweep_dma_bytes_per_cycle | 42.228 | 1108.485 | 0.9021 | 1.0000× | 0.000 / 0.000 |
| sweep_sram_dma_rate | 42.228 | 1108.485 | 0.9021 | 1.0000× | 0.000 / 0.000 |
| sweep_array16_input_bytes_per_cycle | 42.228 | 1108.485 | 0.9021 | 1.0000× | 0.000 / 0.000 |
| sweep_array32_input_bytes_per_cycle | 42.228 | 1108.485 | 0.9021 | 1.0000× | 0.000 / 0.000 |
| sweep_array_output_bytes_per_cycle | 42.192 | 1106.820 | 0.9035 | 1.0015× | 0.014 / 0.020 |
| sweep_vector_bytes_per_cycle | 41.527 | 1104.884 | 0.9051 | 1.0033× | 0.198 / 0.149 |
| sweep_dma_setup | 42.166 | 1099.053 | 0.9099 | 1.0086× | 0.006 / 0.008 |
| sweep_dram_read_start | 42.173 | 1099.299 | 0.9097 | 1.0084× | 0.005 / 0.006 |
| sweep_dram_write_start | 42.224 | 1108.301 | 0.9023 | 1.0002× | 0.001 / 0.001 |
| sweep_array_launch | 41.430 | 1089.456 | 0.9179 | 1.0175× | 0.299 / 0.276 |
| sweep_vector_launch | 42.066 | 1100.845 | 0.9084 | 1.0069× | 0.060 / 0.160 |
| sweep_scratchpad_bytes | 42.228 | 1108.485 | 0.9021 | 1.0000× | 0.000 / 0.000 |
| rowblocked | 48.880 | 1117.732 | 0.8947 | 0.9917× | — |
| independent_heads | 43.649 | 1184.924 | 0.8439 | 0.9355× | — |
| sweep_mac_rate | 40.470 | 1065.578 | 0.9385 | 1.0403× | 0.323 / 0.299 |
| a16exact | 334.828 | 7513.677 | 0.1331 | 0.1475× | — |
| fixedshift | 42.228 | 1108.485 | 0.9021 | 1.0000× | — |

## Batch 16, one variable at a time

| Case | Device TTFT (s) | Decode (ms/step) | Tokens/s/user | Speedup vs baseline | Prefill / decode attention elasticity |
|---|---:|---:|---:|---:|---:|
| baseline | 670.250 | 1526.219 | 0.6552 | 1.0000× | — |
| naive | 686.916 | 1761.844 | 0.5676 | 0.8663× | — |
| exact4 | 2405.509 | 5352.993 | 0.1868 | 0.2851× | — |
| bounded | 670.250 | 1526.219 | 0.6552 | 1.0000× | — |
| saturate | 670.250 | 1526.219 | 0.6552 | 1.0000× | — |
| queued_delivery | 546.494 | 1209.826 | 0.8266 | 1.2615× | — |
| duplex_dma | 670.250 | 1526.219 | 0.6552 | 1.0000× | — |
| transpose_decode | 670.242 | 1417.204 | 0.7056 | 1.0769× | — |
| overlap_drain | 827.377 | 1692.119 | 0.5910 | 0.9020× | — |
| double_buffer | 673.741 | 1647.389 | 0.6070 | 0.9264× | — |
| wavefront_fill | 728.599 | 1615.156 | 0.6191 | 0.9449× | — |
| query_tile | 682.249 | 1526.219 | 0.6552 | 1.0000× | — |
| key_tile | 680.296 | 1662.588 | 0.6015 | 0.9180× | — |
| double_output_port | 667.633 | 1514.304 | 0.6604 | 1.0079× | — |
| double_both_dma_links | 639.708 | 1449.553 | 0.6899 | 1.0529× | — |
| sweep_dma_bytes_per_cycle | 670.250 | 1526.219 | 0.6552 | 1.0000× | 0.000 / 0.000 |
| sweep_sram_dma_rate | 670.250 | 1526.219 | 0.6552 | 1.0000× | 0.000 / 0.000 |
| sweep_array16_input_bytes_per_cycle | 670.250 | 1526.219 | 0.6552 | 1.0000× | 0.000 / 0.000 |
| sweep_array32_input_bytes_per_cycle | 670.250 | 1526.219 | 0.6552 | 1.0000× | 0.000 / 0.000 |
| sweep_array_output_bytes_per_cycle | 669.677 | 1523.613 | 0.6563 | 1.0017× | 0.014 / 0.020 |
| sweep_vector_bytes_per_cycle | 659.028 | 1516.672 | 0.6593 | 1.0063× | 0.198 / 0.149 |
| sweep_dma_setup | 669.290 | 1516.406 | 0.6595 | 1.0065× | 0.006 / 0.008 |
| sweep_dram_read_start | 669.394 | 1516.755 | 0.6593 | 1.0062× | 0.005 / 0.006 |
| sweep_dram_write_start | 670.172 | 1525.958 | 0.6553 | 1.0002× | 0.001 / 0.001 |
| sweep_array_launch | 657.573 | 1496.574 | 0.6682 | 1.0198× | 0.299 / 0.276 |
| sweep_vector_launch | 667.726 | 1512.382 | 0.6612 | 1.0091× | 0.060 / 0.160 |
| sweep_scratchpad_bytes | 670.250 | 1526.219 | 0.6552 | 1.0000× | 0.000 / 0.000 |
| rowblocked | 776.677 | 1674.162 | 0.5973 | 0.9116× | — |
| independent_heads | 692.974 | 2749.237 | 0.3637 | 0.5551× | — |
| sweep_mac_rate | 642.357 | 1471.807 | 0.6794 | 1.0370× | 0.323 / 0.299 |
| a16exact | 5319.327 | 13054.686 | 0.0766 | 0.1169× | — |
| fixedshift | 670.250 | 1526.219 | 0.6552 | 1.0000× | — |

## What changes inside attention

Batch 16, one decoder block. DRAM traffic is read plus write in decimal GB per step. Utilization is the sum of engine busy cycles divided by elapsed attention cycles across the phase. DMA includes first-byte wait and setup. Array service includes input-port stalls. Engines can overlap.

| Mapping | Phase | DRAM GB/step | DMA busy | 16×16 busy | 32×16 busy |
|---|---|---:|---:|---:|---:|
| naive | prefill | 13.6231 | 33.4% | 17.0% | 25.5% |
| naive | decode | 0.1518 | 37.7% | 22.1% | 0.0% |
| rowblocked | prefill | 34.9637 | 23.6% | 24.5% | 0.0% |
| rowblocked | decode | 0.1429 | 23.6% | 25.6% | 0.0% |
| flashattention | prefill | 4.8989 | 12.8% | 25.1% | 37.7% |
| flashattention | decode | 0.1429 | 32.1% | 34.8% | 0.0% |

## Batch × attention × precision

Costs below are per-user tokens/s for the entire model. The precision modes share the same one-pass timing where they transfer the same widths and perform the same number of native operations. The assumed scaling hardware and the software digit decompositions have different numerical contracts.

| B | Precision | Naive | SRAM row-blocked | FlashAttention | CUDA FlashAttention relative RMS, prefill / decode | Thresholds |
|---:|---|---:|---:|---:|---:|---|
| 1 | a16exact | 0.1328 | 0.1329 | 0.1331 | 0.0228% / 0.0272% | pass |
| 1 | bounded | 0.8903 | 0.8947 | 0.9021 | 37.0453% / 44.7058% | outside tolerance |
| 1 | exact4 | 0.2584 | 0.2588 | 0.2594 | 5.2343% / 5.6833% | pass |
| 1 | fixedshift | 0.8903 | 0.8947 | 0.9021 | 5.4762% / 5.9011% | pass |
| 1 | saturate | 0.8903 | 0.8947 | 0.9021 | 63.4195% / 77.1227% | outside tolerance |
| 1 | scaled | 0.8903 | 0.8947 | 0.9021 | 5.2342% / 5.6829% | pass |
| 16 | a16exact | 0.0752 | 0.0757 | 0.0766 | 0.0238% / 0.0202% | pass |
| 16 | bounded | 0.5676 | 0.5973 | 0.6552 | 37.8831% / 38.4107% | outside tolerance |
| 16 | exact4 | 0.1789 | 0.1818 | 0.1868 | 5.3074% / 4.7688% | pass |
| 16 | fixedshift | 0.5676 | 0.5973 | 0.6552 | 5.5015% / 4.8478% | pass |
| 16 | saturate | 0.5676 | 0.5973 | 0.6552 | 64.9489% / 55.7592% | outside tolerance |
| 16 | scaled | 0.5676 | 0.5973 | 0.6552 | 5.3075% / 4.7721% | pass |

## Four accelerators and one host

Local shard service comes from ONNXim. All rates are per user for 16 synchronized users. The configured link is 16 decimal GB/s, aggregate across the host cut. For duplex cases GB/s is per direction. The 70% denominator in this table is each placement’s own compute-only limit. Microbatches share each stage in order. TTFT is the latency until all 16 first tokens are available.

| DP / TP / PP | Microbatches | Host model | Device limit (tokens/s/user) | TTFT with link (s) | Tokens/s/user with link | GB/s for 70% |
|---|---:|---|---:|---:|---:|---:|
| 4 / 1 / 1 | 1 | serial | 0.8390 | 167.833 | 0.8390 | 2.505792e-07 |
| 4 / 1 / 1 | 1 | chunked | 0.8390 | 167.833 | 0.8390 | 1.252896e-07 |
| 4 / 1 / 1 | 1 | overlap_bound | 0.8390 | 167.833 | 0.8390 | 3.758687e-08 |
| 4 / 1 / 1 | 1 | serial_1us | 0.8390 | 167.833 | 0.8390 | 2.505801e-07 |
| 4 / 1 / 1 | 1 | finite_host | 0.8390 | 167.833 | 0.8390 | 2.50585e-07 |
| 1 / 4 / 1 | 1 | serial | 2.2585 | 179.622 | 2.2373 | 0.3536563 |
| 1 / 4 / 1 | 1 | chunked | 2.2585 | 175.327 | 2.2479 | 0.1768287 |
| 1 / 4 / 1 | 1 | overlap_bound | 2.2585 | 171.032 | 2.2585 | 0.0530486 |
| 1 / 4 / 1 | 1 | serial_1us | 2.2585 | 179.622 | 2.2367 | 0.3538988 |
| 1 / 4 / 1 | 1 | finite_host | 2.2585 | 181.340 | 2.2324 | 0.3554901 |
| 1 / 1 / 4 | 1 | serial | 0.6552 | 670.350 | 0.6552 | 0.001202518 |
| 1 / 1 / 4 | 1 | chunked | 0.6552 | 670.300 | 0.6552 | 0.0006012591 |
| 1 / 1 / 4 | 1 | overlap_bound | 0.6552 | 670.250 | 0.6552 | 0.0001803777 |
| 1 / 1 / 4 | 1 | serial_1us | 0.6552 | 670.350 | 0.6552 | 0.001202533 |
| 1 / 1 / 4 | 1 | finite_host | 0.6552 | 670.350 | 0.6552 | 0.001202551 |
| 1 / 1 / 4 | 2 | serial | 0.5928 | 419.684 | 0.5928 | 0.001088028 |
| 1 / 1 / 4 | 2 | chunked | 0.5928 | 419.633 | 0.5928 | 0.0005440141 |
| 1 / 1 / 4 | 2 | overlap_bound | 0.5928 | 419.583 | 0.5928 | 0.0001632042 |
| 1 / 1 / 4 | 2 | serial_1us | 0.5928 | 419.684 | 0.5928 | 0.001088052 |
| 1 / 1 / 4 | 2 | finite_host | 0.5928 | 419.684 | 0.5928 | 0.001088067 |
| 1 / 1 / 4 | 4 | serial | 0.4430 | 293.971 | 0.4429 | 0.0008129526 |
| 1 / 1 / 4 | 4 | chunked | 0.4430 | 293.921 | 0.4429 | 0.0004064763 |
| 1 / 1 / 4 | 4 | overlap_bound | 0.4430 | 293.871 | 0.4430 | 0.0001219429 |
| 1 / 1 / 4 | 4 | serial_1us | 0.4430 | 293.971 | 0.4429 | 0.0008129795 |
| 1 / 1 / 4 | 4 | finite_host | 0.4430 | 293.971 | 0.4429 | 0.0008129879 |
| 2 / 2 / 1 | 1 | serial | 1.4464 | 173.283 | 1.4420 | 0.1132457 |
| 2 / 2 / 1 | 1 | chunked | 1.4464 | 171.135 | 1.4442 | 0.05662294 |
| 2 / 2 / 1 | 1 | overlap_bound | 1.4464 | 168.988 | 1.4464 | 0.01698688 |
| 2 / 2 / 1 | 1 | serial_1us | 1.4464 | 173.283 | 1.4418 | 0.1132954 |
| 2 / 2 / 1 | 1 | finite_host | 1.4464 | 174.314 | 1.4407 | 0.1134921 |
| 2 / 1 / 2 | 1 | serial | 0.7673 | 335.658 | 0.7673 | 0.0004695512 |
| 2 / 1 / 2 | 1 | chunked | 0.7673 | 335.641 | 0.7673 | 0.0002347756 |
| 2 / 1 / 2 | 1 | overlap_bound | 0.7673 | 335.625 | 0.7673 | 7.043269e-05 |
| 2 / 1 / 2 | 1 | serial_1us | 0.7673 | 335.658 | 0.7673 | 0.0004695546 |
| 2 / 1 / 2 | 1 | finite_host | 0.7673 | 335.658 | 0.7673 | 0.000469563 |
| 2 / 1 / 2 | 2 | serial | 0.5476 | 251.815 | 0.5476 | 0.0003351372 |
| 2 / 1 / 2 | 2 | chunked | 0.5476 | 251.798 | 0.5476 | 0.0001675686 |
| 2 / 1 / 2 | 2 | overlap_bound | 0.5476 | 251.782 | 0.5476 | 5.027058e-05 |
| 2 / 1 / 2 | 2 | serial_1us | 0.5476 | 251.815 | 0.5476 | 0.0003351406 |
| 2 / 1 / 2 | 2 | finite_host | 0.5476 | 251.815 | 0.5476 | 0.0003351449 |
| 2 / 1 / 2 | 4 | serial | 0.3384 | 211.180 | 0.3384 | 0.0002071148 |
| 2 / 1 / 2 | 4 | chunked | 0.3384 | 211.163 | 0.3384 | 0.0001035574 |
| 2 / 1 / 2 | 4 | overlap_bound | 0.3384 | 211.146 | 0.3384 | 3.106722e-05 |
| 2 / 1 / 2 | 4 | serial_1us | 0.3384 | 211.180 | 0.3384 | 0.0002071174 |
| 2 / 1 / 2 | 4 | finite_host | 0.3384 | 211.180 | 0.3384 | 0.0002071191 |
| 1 / 2 / 2 | 1 | serial | 1.2440 | 341.783 | 1.2407 | 0.09853666 |
| 1 / 2 / 2 | 1 | chunked | 1.2440 | 339.619 | 1.2423 | 0.04945865 |
| 1 / 2 / 2 | 1 | overlap_bound | 1.2440 | 337.438 | 1.2440 | 0.01483759 |
| 1 / 2 / 2 | 1 | serial_1us | 1.2440 | 341.783 | 1.2405 | 0.09857443 |
| 1 / 2 / 2 | 1 | finite_host | 1.2440 | 342.814 | 1.2397 | 0.09872158 |
| 1 / 2 / 2 | 2 | serial | 0.9298 | 257.855 | 0.9280 | 0.07365072 |
| 1 / 2 / 2 | 2 | chunked | 0.9298 | 255.690 | 0.9289 | 0.03696761 |
| 1 / 2 / 2 | 2 | overlap_bound | 0.9298 | 253.509 | 0.9298 | 0.01109028 |
| 1 / 2 / 2 | 2 | serial_1us | 0.9298 | 257.855 | 0.9277 | 0.07369293 |
| 1 / 2 / 2 | 2 | finite_host | 0.9298 | 258.886 | 0.9273 | 0.07377513 |
| 1 / 2 / 2 | 4 | serial | 0.5875 | 215.751 | 0.5868 | 0.04653554 |
| 1 / 2 / 2 | 4 | chunked | 0.5875 | 213.587 | 0.5871 | 0.02335765 |
| 1 / 2 / 2 | 4 | overlap_bound | 0.5875 | 211.406 | 0.5875 | 0.007007295 |
| 1 / 2 / 2 | 4 | serial_1us | 0.5875 | 215.752 | 0.5866 | 0.04656925 |
| 1 / 2 / 2 | 4 | finite_host | 0.5875 | 216.783 | 0.5864 | 0.04660206 |

### Ideal four-device resource ceiling

Perfect partition, ideal utilization and unbounded issue give 21.213 tokens/s/user (339.41 aggregate). At 70% of this ceiling, TP4 host-star traffic requires at least 0.498 GB/s **each way** with ideal overlap, or 3.322 GB/s of **shared** payload with serialized upload/download and compute. Even infinite host bandwidth leaves the tested TP4 schedule below 70% of this silicon roof.
