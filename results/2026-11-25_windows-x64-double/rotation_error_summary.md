# Rotation error results

Source: `bench_with_reference.csv`. Each value is the median across data seeds, using one
blocked tile-32 result per seed. Relative and absolute errors are summarized separately.

| scales | K | N | M | a | seeds | relative direct | relative rotated | max abs direct | max abs rotated |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| double | 256 | 256 | 1 | 10 | 0;1;2 | 0.236336 | 0.163013 | 12.6948 | 8.7905 |
| double | 256 | 256 | 8 | 1 | 0;1;2 | 0.179864 | 0.181141 | 10.3484 | 11.5725 |
| double | 256 | 256 | 8 | 10 | 0;1;2 | 0.377056 | 0.174251 | 44.8017 | 16.968 |
| double | 256 | 256 | 8 | 100 | 0;1;2 | 0.177360 | 0.147416 | 107.077 | 120.95 |
| double | 256 | 256 | 32 | 10 | 0;1;2 | 0.394295 | 0.173244 | 45.9216 | 18.7232 |
| double | 512 | 512 | 1 | 10 | 0;1;2 | 0.479399 | 0.179452 | 70.9677 | 27.548 |
| double | 512 | 512 | 8 | 1 | 0;1;2 | 0.191825 | 0.187135 | 15.9793 | 15.8631 |
| double | 512 | 512 | 8 | 10 | 0;1;2 | 0.459842 | 0.180605 | 81.3556 | 29.7764 |
| double | 512 | 512 | 8 | 100 | 0;1;2 | 0.174596 | 0.163289 | 219.17 | 216.334 |
| double | 512 | 512 | 32 | 10 | 0;1;2 | 0.471685 | 0.183698 | 81.3556 | 29.7764 |
| double | 1024 | 1024 | 1 | 10 | 0;1;2 | 0.602512 | 0.195751 | 98.7003 | 31.5911 |
| double | 1024 | 1024 | 8 | 1 | 0;1;2 | 0.200399 | 0.201204 | 25.8876 | 28.5535 |
| double | 1024 | 1024 | 8 | 10 | 0;1;2 | 0.558300 | 0.197570 | 100.892 | 35.4238 |
| double | 1024 | 1024 | 8 | 100 | 0;1;2 | 0.194932 | 0.180907 | 259.257 | 257.389 |
| double | 1024 | 1024 | 32 | 10 | 0;1;2 | 0.547260 | 0.196234 | 108.525 | 44.4081 |

At a=100, outlier contributions dominate the FP64 output norm. A lower relative
error alone does not imply a lower maximum absolute error; compare both metrics.
