# HNM progress - threshold 0.5 (frozen 6B + DeepSet)

FP significance: Welch t-test of per-seed FP vs round 0 (n=5 seeds). Round 4 shown for record but excluded from the presentation plot (mining saturated).

| round | n_train | mined_total | FP rate (ensemble) | recall (ensemble) | test MCC | FP vs r0 |
|---:|---:|---:|---:|---:|---:|:--:|
| 0 | 4896 | 0 | 0.0754 | 0.9754 | 0.8503 | — |
| 1 | 5196 | 300 | 0.0514 | 0.9649 | 0.8751 | ** (p=0.009) |
| 2 | 5496 | 600 | 0.0304 | 0.9333 | 0.8772 | *** (p=0.000) |
| 3 | 5549 | 653 | 0.0353 | 0.9368 | 0.8494 | *** (p=0.000) |
| 4 | 5558 | 662 | 0.0448 | 0.9509 | 0.8672 | * (p=0.015) |
