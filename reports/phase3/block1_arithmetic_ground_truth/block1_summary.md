# Phase 3 Block 1 — Arithmetic-basket ground truth

Parameters: S0=100, K=100, r=0.05, sigma=0.2, T=1, d=100, dtype=float64, seed=5101.

Target: discounted arithmetic-average call payoff. Control: discounted geometric-average call payoff with its analytic expectation.

| rho | Y0_MC+CV | 95% CI | relative half-width | M | time (s) | status |
|---:|---:|:---|---:|---:|---:|:---:|
| 0.0 | 4.8814205200 | [4.8807039771, 4.8821370628] | 0.01467898% | 1,000,000 | 0.743 | pass |
| 0.3 | 7.1810171177 | [7.1799878436, 7.1820463917] | 0.01433326% | 1,000,000 | 0.861 | pass |
| 0.5 | 8.3038804536 | [8.3031377497, 8.3046231575] | 0.00894406% | 1,000,000 | 0.865 | pass |

## Control-variate diagnostics

| rho | beta | raw MC SE | CV SE | variance-reduction factor | geometric control price |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.0649580579 | 0.0020087197 | 0.0003655831 | 30.1902 | 2.9718541960 |
| 0.3 | 1.0625515519 | 0.0084588365 | 0.0005251398 | 259.4606 | 6.2561513686 |
| 0.5 | 1.0380482363 | 0.0105913178 | 0.0003789305 | 781.2333 | 7.6622218621 |
