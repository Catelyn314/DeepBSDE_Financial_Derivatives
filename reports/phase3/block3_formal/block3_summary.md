# Phase 3 Block 3 — Formal time-to-accuracy scan

Primary metric: first-hit wall-clock time. Settling is reported separately.

Methodology caveat: the MC+CV benchmark measures the same estimator's internal convergence toward its own high-M reference. It is not an independent-method validation and raw speed ratios versus Deep BSDE should not be presented as an unconditional fair-method contest.

| d | method | target | time | step/paths | actual error | status |
|---:|:---|---:|---:|---:|---:|:---:|
| 10 | Deep BSDE | <0.5% | 19.000000 s | 575 | 0.35076919% | reached |
| 10 | Deep BSDE | <0.2% | 21.000000 s | 775 | 0.06692078% | reached |
| 10 | MC+CV | <0.5% | 0.000238 s | 1000 | 0.19561572% | reached |
| 10 | MC+CV | <0.2% | 0.000238 s | 1000 | 0.19561572% | reached |
| 50 | Deep BSDE | <0.5% | 19.000000 s | 450 | 0.34013586% | reached |
| 50 | Deep BSDE | <0.2% | 19.000000 s | 475 | 0.15425872% | reached |
| 50 | MC+CV | <0.5% | 0.000482 s | 1000 | 0.13068540% | reached |
| 50 | MC+CV | <0.2% | 0.000482 s | 1000 | 0.13068540% | reached |
| 100 | Deep BSDE | <0.5% | 21.000000 s | 475 | 0.36115471% | reached |
| 100 | Deep BSDE | <0.2% | 22.000000 s | 525 | 0.07824526% | reached |
| 100 | MC+CV | <0.5% | 0.000859 s | 1000 | 0.04868231% | reached |
| 100 | MC+CV | <0.2% | 0.000859 s | 1000 | 0.04868231% | reached |

No MC+CV cell hit the 2e7 path cap.

See `block3_bsde_stability_diagnostic.csv` for settling results; they are not used as the primary time-to-accuracy metric.
