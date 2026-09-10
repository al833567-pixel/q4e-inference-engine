# GGUF shard 2-4 tensor inventory — breakdown (stepA_h1h2 / s2_inventory)

- Files: 3 shards of `00002` (shards 2, 3, 4 of 4)
- String-length convention used: shard 00002 -> `u64`; shard 00003 -> `u64`; shard 00004 -> `u64`
- Total tensors: **1224** (expected 1224)

## Per-shard summary

| shard | convention | tensor_count | header_bytes | first_data_offset | offset_mode | sum_tensor_bytes | file_size | residual |
|---|---|---|---|---|---|---|---|---|
| 00002 | u64 | 297 | 18695 | 0 | relative(first=0) | 49859564441 | 49859583136 | 0 |
| 00003 | u64 | 752 | 48045 | 0 | relative(first=0) | 49376093459 | 49376141504 | 0 |
| 00004 | u64 | 175 | 11267 | 0 | relative(first=0) | 12087972253 | 12087983520 | 0 |

Grand total tensor bytes: **111323630153** = 106166.49 MiB = 103.6782 GiB

## Group table (regex R = `^blk\.\d+\.ffn_(up|down|gate|gate_up)_(ch|)exps`, SEARCH)

| group | rule | tensors | bytes | MiB |
|---|---|---|---|---|
| G1 | R matches, layer 0..37 | 114 | 60764979200 | 57950.00 |
| G2 | R matches, layer 38..47 | 30 | 16252928000 | 15500.00 |
| G3 | name contains 'shexp' | 192 | 251166745 | 239.53 |
| G4 | name contains 'ple' or 'per_layer' | 7 | 28835240960 | 27499.43 |
| G5 | everything else | 881 | 5219315248 | 4977.53 |

## VRAM predictions (K=10, ngl=all)

- **PREDICTED_VRAM_K10** = grand_total - G1 = 111323630153 - 60764979200 = **50558650953** bytes = **48216.49 MiB** = **47.0864 GiB**
- **PREDICTED_VRAM_K10_PLE_CPU** = grand_total - G1 - G4 = 111323630153 - 60764979200 - 28835240960 = **21723409993** bytes = **20717.06 MiB** = **20.2315 GiB**

## Per-layer routed-expert bytes (sum of G1/G2 tensors of that layer)

| layer | expert tensors | bytes | MiB |
|---|---|---|---|
| 0 | 3 | 1572864000 | 1500.00 |
| 1 | 3 | 1572864000 | 1500.00 |
| 37 | 3 | 1572864000 | 1500.00 |
| 38 | 3 | 1572864000 | 1500.00 |
| 47 | 3 | 1835008000 | 1750.00 |

## 10 largest single tensors

| # | name | shard | MiB |
|---|---|---|---|
| 1 | per_layer_token_embd.weight | 00002 | 27465.95 |
| 2 | blk.2.ffn_down_exps.weight | 00002 | 850.00 |
| 3 | blk.4.ffn_down_exps.weight | 00002 | 850.00 |
| 4 | blk.30.ffn_down_exps.weight | 00003 | 850.00 |
| 5 | blk.46.ffn_down_exps.weight | 00004 | 850.00 |
| 6 | blk.47.ffn_down_exps.weight | 00004 | 850.00 |
| 7 | output.weight | 00002 | 644.14 |
| 8 | token_embd.weight | 00002 | 644.14 |
| 9 | blk.0.ffn_down_exps.weight | 00002 | 600.00 |
| 10 | blk.1.ffn_down_exps.weight | 00002 | 600.00 |
