# Phase 3 Block 4 — Delta (Z0) comparison

Selected bump: 1.000000.
d=10 candidate RMSE: 1.7107981276e-05; mean-absolute Delta ratio: 0.999787.

The network output is the BSDE integrand Z0. For dimensional comparison with bump Delta, Delta=Z0/(sigma*S0)=Z0/20.

## Difference distribution

- RMSE: 5.4621806328e-04
- MAE: 4.1906042570e-04
- Bias (BSDE - MC): -2.1204439856e-05
- Maximum absolute difference: 1.6001080380e-03
- Component correlation: 0.10802427

## Iso-accuracy timing

| method | precision | work | time |
|:---|---:|---:|---:|
| Deep BSDE | RMS SEM=5.6112845903e-04 | 3 seeds | 438.259 s |
| MC+CV CRN bump | RMS SE=6.7895455175e-05 | M=1000 | 0.002222 s |
| MC+CV CRN high-precision reference | diagnostic | M=1000000 | 1.978232 s |

Each MC run evaluates the baseline plus 2x100 bumped prices (201 valuations), vectorized over one shared CRN path set. CRN is used for every plus/minus pair. Pathwise differentiation is not used because the payoff is non-differentiable at the kink avg(S)=K, where its variance can become unstable.
