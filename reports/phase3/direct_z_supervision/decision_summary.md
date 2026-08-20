# Phase 3 direct-Z supervision — decision summary

## Outcome

The preregistered 1D experiment was completed through the one-time final gate.
The selected candidate was `lambda_z=100` with all 25 nonterminal left
endpoints supervised. It improved Delta and replication performance materially,
but did **not** pass every preregistered gate. The 100D upgrade is therefore not
authorized.

## Preflight

- Analytic Delta versus central finite difference: PASS, maximum absolute error
  `2.7091482391e-09` (threshold `1e-7`).
- `Z* = sigma S Delta*`: PASS.
- Extracted Z versus the values used by the forward pass: PASS, exact equality.
- Fixed-batch `lambda_z=0` forward, loss, and one-step update parity: PASS,
  exact equality.
- Full 6000-step `lambda_z=0` canary: PASS; maximum step/loss/Y0 history
  discrepancy `4.9609241159e-09`, saved variables exactly equal.
- Analytic-Z synthetic loss: PASS, exactly zero.
- Random-five reproducibility and no-replacement check: PASS.
- Training/development/final path-hash isolation: PASS.

## Development selection

| lambda_z | median all-time Delta RMSE | mean replication MSE |
|---:|---:|---:|
| 0.1 | 13.9673% | 7.4269 |
| 1 | 13.4906% | 7.3071 |
| 10 | 13.3834% | 7.0772 |
| 100 | **11.2476%** | **5.5539** |

All candidates satisfied the development Y0 constraint. `lambda_z=100` was
selected by the preregistered median-Delta rule.

| time scheme at lambda_z=100 | median all-time Delta RMSE | mean replication MSE |
|---|---:|---:|
| all-time | **11.2476%** | **5.5539** |
| random-five | 17.5001% | 9.6863 |
| near-t0 | 12.9539% | 6.7415 |

The final candidate was locked as `lambda_z=100`, all-time, using development
seed 6002 only.

## One-time final gate (seed 6001, 512 paths)

| train seed | Y0 error | Delta RMSE | improved times | replication MSE | A | B | C | D | E |
|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 5101 | 0.05993% | 11.6026% | 25/25 | 5.1163 | PASS | **FAIL** | PASS | **FAIL** | PASS |
| 5201 | 0.01836% | 10.9438% | 24/25 | 4.9490 | PASS | PASS | PASS | **FAIL** | PASS |
| 5202 | 0.02099% | 11.3482% | 25/25 | 4.9934 | PASS | **FAIL** | PASS | **FAIL** | PASS |

Gate B failed because seeds 5101 and 5202 did not reach their strict 20%
relative-improvement thresholds (11.0693% and 11.1943%). Gate D failed because
absolute pooled-bias reductions were approximately 15.7%, 18.5%, and 18.4%,
short of the required 20%; their pooled slopes nevertheless moved closer to 1.

Replication improved by more than 10% for every seed, with final MSE ratios to
the analytic-Z benchmark of 1.946, 1.882, and 1.899 and excess-gap closures of
34.8%, 40.7%, and 38.2%.

## Decision

**FAIL 1D Final Gate; stop before 100D.** The preregistered thresholds remain
unchanged. Further work should diagnose the residual systematic negative Delta
bias and the seed-dependent shortfall against Gate B rather than migrate this
candidate to the geometric basket.
