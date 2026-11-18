# Four accelerators on one host

The 16 users start together with 2,048 tokens each. The first pass produces one token per user. The following 255 cached steps each produce 16 tokens. TTFT ends when the last of the 16 first tokens reaches the modeled host. Each device already holds its weights. Local stage service comes from a native ONNXim rank schedule for each visible context. Host transfers and reductions are analytical. [Raw per-case metrics](../example/arago/results/four-case-metrics.json)

## Parallelism and bandwidth

| DP/TP/PP | Work assigned to each rank |
|---|---|
| 4/1/1 | Four independent replicas each serve four users through 32 layers. Only IDs cross the host link. |
| 1/4/1 | All users share four tensor shards. Each rank owns eight Q heads, two KV heads and a quarter of each block projection. Two host reductions per layer combine the row-sharded projections. |
| 1/1/4 | Each rank owns eight consecutive layers. Activations pass through the host at three stage boundaries. |
| 2/2/1 | Two replicas each serve eight users across two tensor shards. |
| 2/1/2 | Two replicas each serve eight users through two 16-layer stages. |
| 1/2/2 | Each of two 16-layer stages uses a pair of tensor shards. At the stage boundary a representative uploads the hidden state and the host forwards it to both next-stage ranks. |

Pipeline experiments split a synchronized cohort into 1, 2 or 4 equal microbatches. Within each stage the microbatches serialize; stages can process different microbatches in parallel. The host width is an aggregate cut across all four devices. Full duplex quotes GB/s per direction; serialized transport quotes shared GB/s. Decoder blocks use W8/A16 traffic and the baseline scaled native return. Device vocabulary heads are replicated across tensor ranks.

The common four-device ideal ceiling is **21.213 tokens/s/user** (**339.41 aggregate**). It assumes 4 × 768 GMAC/s, 4 × 64 GB/s, one weight fetch, ideal KV reuse, perfect partition and resource overlap. Reaching 70% means **14.849 tokens/s/user**. On the hypothetical ideal TP4 mapping the host payload alone needs **0.498 GB/s per direction** under complete overlap, or **3.322 GB/s shared** with serialized transfer and compute. The fastest mapped device schedule is **2.259 tokens/s/user**, so every tested schedule misses the ideal 70% target even at infinite host bandwidth.

The second practical comparison is 70% of the **fastest mapped schedule** (1.581 tokens/s/user). The third comparison is 70% of **each row’s own device limit**. A dash means the requested target exceeds the modeled infinite-link rate. The own-limit and best-limit bandwidths include that row’s fixed startup and host costs.

## All four-device cases

Case names give DP/TP/PP, microbatch count, and host model. TBT is mean time between successive tokens for one user. Aggregate tokens/s sums all 16 users. The common ideal 70% bandwidth answer for every row is unreachable. Columns for the two attainable-target calculations use decimal GB/s.

| Case | TTFT (s) | TBT (ms) | Tokens/s/user | Tokens/s total | Device limit /user | GB/s for 70% own | GB/s for 70% best |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4/1/1 · 1 · serial | 167.833 | 1191.905 | 0.8390 | 13.424 | 0.8390 | 2.50579e-07 | — |
| 4/1/1 · 1 · chunked | 167.833 | 1191.905 | 0.8390 | 13.424 | 0.8390 | 1.2529e-07 | — |
| 4/1/1 · 1 · overlap_bound | 167.833 | 1191.905 | 0.8390 | 13.424 | 0.8390 | 3.75869e-08 | — |
| 4/1/1 · 1 · serial_1us | 167.833 | 1191.907 | 0.8390 | 13.424 | 0.8390 | 2.5058e-07 | — |
| 4/1/1 · 1 · finite_host | 167.833 | 1191.917 | 0.8390 | 13.424 | 0.8390 | 2.50585e-07 | — |
| 1/4/1 · 1 · serial | 179.622 | 446.963 | 2.2373 | 35.797 | 2.2585 | 0.353656 | 0.353656 |
| 1/4/1 · 1 · chunked | 175.327 | 444.866 | 2.2479 | 35.966 | 2.2585 | 0.176829 | 0.176829 |
| 1/4/1 · 1 · overlap_bound | 171.032 | 442.769 | 2.2585 | 36.136 | 2.2585 | 0.0530486 | 0.0530486 |
| 1/4/1 · 1 · serial_1us | 179.622 | 447.093 | 2.2367 | 35.787 | 2.2585 | 0.353899 | 0.353899 |
| 1/4/1 · 1 · finite_host | 181.340 | 447.942 | 2.2324 | 35.719 | 2.2585 | 0.35549 | 0.35549 |
| 1/1/4 · 1 · serial | 670.350 | 1526.269 | 0.6552 | 10.483 | 0.6552 | 0.00120252 | — |
| 1/1/4 · 1 · chunked | 670.300 | 1526.244 | 0.6552 | 10.483 | 0.6552 | 0.000601259 | — |
| 1/1/4 · 1 · overlap_bound | 670.250 | 1526.219 | 0.6552 | 10.483 | 0.6552 | 0.000180378 | — |
| 1/1/4 · 1 · serial_1us | 670.350 | 1526.277 | 0.6552 | 10.483 | 0.6552 | 0.00120253 | — |
| 1/1/4 · 1 · finite_host | 670.350 | 1526.287 | 0.6552 | 10.483 | 0.6552 | 0.00120255 | — |
| 1/1/4 · 2 · serial | 419.684 | 1686.868 | 0.5928 | 9.485 | 0.5928 | 0.00108803 | — |
| 1/1/4 · 2 · chunked | 419.633 | 1686.844 | 0.5928 | 9.485 | 0.5928 | 0.000544014 | — |
| 1/1/4 · 2 · overlap_bound | 419.583 | 1686.819 | 0.5928 | 9.485 | 0.5928 | 0.000163204 | — |
| 1/1/4 · 2 · serial_1us | 419.684 | 1686.884 | 0.5928 | 9.485 | 0.5928 | 0.00108805 | — |
| 1/1/4 · 2 · finite_host | 419.684 | 1686.894 | 0.5928 | 9.485 | 0.5928 | 0.00108807 | — |
| 1/1/4 · 4 · serial | 293.971 | 2257.631 | 0.4429 | 7.087 | 0.4430 | 0.000812953 | — |
| 1/1/4 · 4 · chunked | 293.921 | 2257.606 | 0.4429 | 7.087 | 0.4430 | 0.000406476 | — |
| 1/1/4 · 4 · overlap_bound | 293.871 | 2257.581 | 0.4430 | 7.087 | 0.4430 | 0.000121943 | — |
| 1/1/4 · 4 · serial_1us | 293.971 | 2257.663 | 0.4429 | 7.087 | 0.4430 | 0.000812979 | — |
| 1/1/4 · 4 · finite_host | 293.971 | 2257.673 | 0.4429 | 7.087 | 0.4430 | 0.000812988 | — |
| 2/2/1 · 1 · serial | 173.283 | 693.462 | 1.4420 | 23.073 | 1.4464 | 0.113246 | — |
| 2/2/1 · 1 · chunked | 171.135 | 692.414 | 1.4442 | 23.108 | 1.4464 | 0.0566229 | — |
| 2/2/1 · 1 · overlap_bound | 168.988 | 691.365 | 1.4464 | 23.143 | 1.4464 | 0.0169869 | — |
| 2/2/1 · 1 · serial_1us | 173.283 | 693.592 | 1.4418 | 23.068 | 1.4464 | 0.113295 | — |
| 2/2/1 · 1 · finite_host | 174.314 | 694.106 | 1.4407 | 23.051 | 1.4464 | 0.113492 | — |
| 2/1/2 · 1 · serial | 335.658 | 1303.320 | 0.7673 | 12.276 | 0.7673 | 0.000469551 | — |
| 2/1/2 · 1 · chunked | 335.641 | 1303.312 | 0.7673 | 12.276 | 0.7673 | 0.000234776 | — |
| 2/1/2 · 1 · overlap_bound | 335.625 | 1303.304 | 0.7673 | 12.276 | 0.7673 | 7.04327e-05 | — |
| 2/1/2 · 1 · serial_1us | 335.658 | 1303.324 | 0.7673 | 12.276 | 0.7673 | 0.000469555 | — |
| 2/1/2 · 1 · finite_host | 335.658 | 1303.334 | 0.7673 | 12.276 | 0.7673 | 0.000469563 | — |
| 2/1/2 · 2 · serial | 251.815 | 1826.039 | 0.5476 | 8.762 | 0.5476 | 0.000335137 | — |
| 2/1/2 · 2 · chunked | 251.798 | 1826.031 | 0.5476 | 8.762 | 0.5476 | 0.000167569 | — |
| 2/1/2 · 2 · overlap_bound | 251.782 | 1826.022 | 0.5476 | 8.762 | 0.5476 | 5.02706e-05 | — |
| 2/1/2 · 2 · serial_1us | 251.815 | 1826.047 | 0.5476 | 8.762 | 0.5476 | 0.000335141 | — |
| 2/1/2 · 2 · finite_host | 251.815 | 1826.057 | 0.5476 | 8.762 | 0.5476 | 0.000335145 | — |
| 2/1/2 · 4 · serial | 211.180 | 2954.744 | 0.3384 | 5.415 | 0.3384 | 0.000207115 | — |
| 2/1/2 · 4 · chunked | 211.163 | 2954.736 | 0.3384 | 5.415 | 0.3384 | 0.000103557 | — |
| 2/1/2 · 4 · overlap_bound | 211.146 | 2954.728 | 0.3384 | 5.415 | 0.3384 | 3.10672e-05 | — |
| 2/1/2 · 4 · serial_1us | 211.180 | 2954.760 | 0.3384 | 5.415 | 0.3384 | 0.000207117 | — |
| 2/1/2 · 4 · finite_host | 211.180 | 2954.770 | 0.3384 | 5.415 | 0.3384 | 0.000207119 | — |
| 1/2/2 · 1 · serial | 341.783 | 806.001 | 1.2407 | 19.851 | 1.2440 | 0.0985367 | — |
| 1/2/2 · 1 · chunked | 339.619 | 804.945 | 1.2423 | 19.877 | 1.2440 | 0.0494586 | — |
| 1/2/2 · 1 · overlap_bound | 337.438 | 803.880 | 1.2440 | 19.903 | 1.2440 | 0.0148376 | — |
| 1/2/2 · 1 · serial_1us | 341.783 | 806.133 | 1.2405 | 19.848 | 1.2440 | 0.0985744 | — |
| 1/2/2 · 1 · finite_host | 342.814 | 806.647 | 1.2397 | 19.835 | 1.2440 | 0.0987216 | — |
| 1/2/2 · 2 · serial | 257.855 | 1077.626 | 0.9280 | 14.847 | 0.9298 | 0.0736507 | — |
| 1/2/2 · 2 · chunked | 255.690 | 1076.569 | 0.9289 | 14.862 | 0.9298 | 0.0369676 | — |
| 1/2/2 · 2 · overlap_bound | 253.509 | 1075.504 | 0.9298 | 14.877 | 0.9298 | 0.0110903 | — |
| 1/2/2 · 2 · serial_1us | 257.855 | 1077.890 | 0.9277 | 14.844 | 0.9298 | 0.0736929 | — |
| 1/2/2 · 2 · finite_host | 258.886 | 1078.403 | 0.9273 | 14.837 | 0.9298 | 0.0737751 | — |
| 1/2/2 · 4 · serial | 215.751 | 1704.297 | 0.5868 | 9.388 | 0.5875 | 0.0465355 | — |
| 1/2/2 · 4 · chunked | 213.587 | 1703.240 | 0.5871 | 9.394 | 0.5875 | 0.0233577 | — |
| 1/2/2 · 4 · overlap_bound | 211.406 | 1702.175 | 0.5875 | 9.400 | 0.5875 | 0.0070073 | — |
| 1/2/2 · 4 · serial_1us | 215.752 | 1704.825 | 0.5866 | 9.385 | 0.5875 | 0.0465692 | — |
| 1/2/2 · 4 · finite_host | 216.783 | 1705.338 | 0.5864 | 9.382 | 0.5875 | 0.0466021 | — |

## Host critical path

At nominal 16 GB/s, serial pays both directions. Chunked allows simultaneous directional transfers; overlap_bound also permits complete transfer/device overlap. Serial_1us adds 1 µs per host communication phase. Finite_host adds host reduction at 50 GB/s plus 10 µs dispatch per step. These host options are assumptions. Each row below adds rank-local service, exposed link time, startup and host work to recover its TBT. For overlap, exposed link time is max(wire time − local time, 0) evaluated per cached step. The signed elasticity is the fractional change in full decode time when increasing host bandwidth by 10%, divided by 0.10. A value near zero means additional host bandwidth buys little under this scenario. The separate [per-case kernel sweep](case-metrics.md) identifies the most elastic accelerator parameter for each rank-local attention kernel.

| Case | Local (ms) | Exposed link (ms) | Link startup (ms) | CPU service (ms) | Host bandwidth elasticity |
|---|---:|---:|---:|---:|---:|
| 4/1/1 · 1 · serial | 1191.905 | 0.0000080 | 0.0000 | 0.0000 | -0.00000001 |
| 4/1/1 · 1 · chunked | 1191.905 | 0.0000040 | 0.0000 | 0.0000 | -0.00000000 |
| 4/1/1 · 1 · overlap_bound | 1191.905 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 4/1/1 · 1 · serial_1us | 1191.905 | 0.0000080 | 0.0020 | 0.0000 | -0.00000001 |
| 4/1/1 · 1 · finite_host | 1191.905 | 0.0000080 | 0.0020 | 0.0100 | -0.00000001 |
| 1/4/1 · 1 · serial | 442.769 | 4.1943240 | 0.0000 | 0.0000 | -0.00853095 |
| 1/4/1 · 1 · chunked | 442.769 | 2.0971680 | 0.0000 | 0.0000 | -0.00428559 |
| 1/4/1 · 1 · overlap_bound | 442.769 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 1/4/1 · 1 · serial_1us | 442.769 | 4.1943240 | 0.1300 | 0.0000 | -0.00852847 |
| 1/4/1 · 1 · finite_host | 442.769 | 4.1943240 | 0.1300 | 0.8489 | -0.00851231 |
| 1/1/4 · 1 · serial | 1526.219 | 0.0491600 | 0.0000 | 0.0000 | -0.00002928 |
| 1/1/4 · 1 · chunked | 1526.219 | 0.0245800 | 0.0000 | 0.0000 | -0.00001464 |
| 1/1/4 · 1 · overlap_bound | 1526.219 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 1/1/4 · 1 · serial_1us | 1526.219 | 0.0491600 | 0.0080 | 0.0000 | -0.00002928 |
| 1/1/4 · 1 · finite_host | 1526.219 | 0.0491600 | 0.0080 | 0.0100 | -0.00002928 |
| 1/1/4 · 2 · serial | 1686.819 | 0.0491600 | 0.0000 | 0.0000 | -0.00002649 |
| 1/1/4 · 2 · chunked | 1686.819 | 0.0245800 | 0.0000 | 0.0000 | -0.00001325 |
| 1/1/4 · 2 · overlap_bound | 1686.819 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 1/1/4 · 2 · serial_1us | 1686.819 | 0.0491600 | 0.0160 | 0.0000 | -0.00002649 |
| 1/1/4 · 2 · finite_host | 1686.819 | 0.0491600 | 0.0160 | 0.0100 | -0.00002649 |
| 1/1/4 · 4 · serial | 2257.581 | 0.0491600 | 0.0000 | 0.0000 | -0.00001980 |
| 1/1/4 · 4 · chunked | 2257.581 | 0.0245800 | 0.0000 | 0.0000 | -0.00000990 |
| 1/1/4 · 4 · overlap_bound | 2257.581 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 1/1/4 · 4 · serial_1us | 2257.581 | 0.0491600 | 0.0320 | 0.0000 | -0.00001980 |
| 1/1/4 · 4 · finite_host | 2257.581 | 0.0491600 | 0.0320 | 0.0100 | -0.00001980 |
| 2/2/1 · 1 · serial | 691.365 | 2.0971640 | 0.0000 | 0.0000 | -0.00274927 |
| 2/2/1 · 1 · chunked | 691.365 | 1.0485840 | 0.0000 | 0.0000 | -0.00137672 |
| 2/2/1 · 1 · overlap_bound | 691.365 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 2/2/1 · 1 · serial_1us | 691.365 | 2.0971640 | 0.1300 | 0.0000 | -0.00274875 |
| 2/2/1 · 1 · finite_host | 691.365 | 2.0971640 | 0.1300 | 0.5133 | -0.00274672 |
| 2/1/2 · 1 · serial | 1303.304 | 0.0163920 | 0.0000 | 0.0000 | -0.00001143 |
| 2/1/2 · 1 · chunked | 1303.304 | 0.0081960 | 0.0000 | 0.0000 | -0.00000572 |
| 2/1/2 · 1 · overlap_bound | 1303.304 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 2/1/2 · 1 · serial_1us | 1303.304 | 0.0163920 | 0.0040 | 0.0000 | -0.00001143 |
| 2/1/2 · 1 · finite_host | 1303.304 | 0.0163920 | 0.0040 | 0.0100 | -0.00001143 |
| 2/1/2 · 2 · serial | 1826.022 | 0.0163920 | 0.0000 | 0.0000 | -0.00000816 |
| 2/1/2 · 2 · chunked | 1826.022 | 0.0081960 | 0.0000 | 0.0000 | -0.00000408 |
| 2/1/2 · 2 · overlap_bound | 1826.022 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 2/1/2 · 2 · serial_1us | 1826.022 | 0.0163920 | 0.0080 | 0.0000 | -0.00000816 |
| 2/1/2 · 2 · finite_host | 1826.022 | 0.0163920 | 0.0080 | 0.0100 | -0.00000816 |
| 2/1/2 · 4 · serial | 2954.728 | 0.0163920 | 0.0000 | 0.0000 | -0.00000504 |
| 2/1/2 · 4 · chunked | 2954.728 | 0.0081960 | 0.0000 | 0.0000 | -0.00000252 |
| 2/1/2 · 4 · overlap_bound | 2954.728 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 2/1/2 · 4 · serial_1us | 2954.728 | 0.0163920 | 0.0160 | 0.0000 | -0.00000504 |
| 2/1/2 · 4 · finite_host | 2954.728 | 0.0163920 | 0.0160 | 0.0100 | -0.00000504 |
| 1/2/2 · 1 · serial | 803.880 | 2.1217400 | 0.0000 | 0.0000 | -0.00239312 |
| 1/2/2 · 1 · chunked | 803.880 | 1.0649680 | 0.0000 | 0.0000 | -0.00120276 |
| 1/2/2 · 1 · overlap_bound | 803.880 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 1/2/2 · 1 · serial_1us | 803.880 | 2.1217400 | 0.1320 | 0.0000 | -0.00239272 |
| 1/2/2 · 1 · finite_host | 803.880 | 2.1217400 | 0.1320 | 0.5133 | -0.00239120 |
| 1/2/2 · 2 · serial | 1075.504 | 2.1217400 | 0.0000 | 0.0000 | -0.00178991 |
| 1/2/2 · 2 · chunked | 1075.504 | 1.0649680 | 0.0000 | 0.0000 | -0.00089929 |
| 1/2/2 · 2 · overlap_bound | 1075.504 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 1/2/2 · 2 · serial_1us | 1075.504 | 2.1217400 | 0.2640 | 0.0000 | -0.00178947 |
| 1/2/2 · 2 · finite_host | 1075.504 | 2.1217400 | 0.2640 | 0.5133 | -0.00178862 |
| 1/2/2 · 4 · serial | 1702.175 | 2.1217400 | 0.0000 | 0.0000 | -0.00113176 |
| 1/2/2 · 4 · chunked | 1702.175 | 1.0649680 | 0.0000 | 0.0000 | -0.00056842 |
| 1/2/2 · 4 · overlap_bound | 1702.175 | 0.0000000 | 0.0000 | 0.0000 | 0.00000000 |
| 1/2/2 · 4 · serial_1us | 1702.175 | 2.1217400 | 0.5280 | 0.0000 | -0.00113141 |
| 1/2/2 · 4 · finite_host | 1702.175 | 2.1217400 | 0.5280 | 0.5133 | -0.00113107 |

## Decisions supported by the comparisons

- **Favor tensor parallelism for steady decode with this host.** TP4 achieves 2.237 tokens/s/user versus 0.839 for DP4 at 16 GB/s shared, a 2.67× gain. DP4 returns the synchronized first tokens sooner (167.8 versus 179.6 s).
- **If four pipeline stages are required, choose the microbatch count for the latency goal.** Moving from four microbatches to one improves steady interactivity from 0.443 to 0.655. Moving in the other direction improves TTFT from 670.4 to 294.0 s. The skinny decode projections lose utilization while prefilling can keep more stages busy.
- **Prioritize rank service over faster links at 16 GB/s for TP4.** Serial link exposure is 4.194 ms per token against 442.77 ms of local work. Chunking the two directions raises rate to 2.248; ideal compute/transfer overlap reaches 2.259. Host-mediated dependencies may prevent realizing that last bound.

To reproduce, run `python3 tools/arago/four_case_metrics.py` after building the focused target. The program replays all 256 iterations for each placement/microbatch and solves each host crossing at its absolute tokens/s target. Controls and raw directional byte counts are in the JSON.
