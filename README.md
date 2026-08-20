# DeepBSDE for Financial Derivatives

A TensorFlow/Keras research implementation of the Deep BSDE method for pricing financial derivatives and solving high-dimensional parabolic PDEs.

This project extends the original [DeepBSDE implementation](https://github.com/frankhan91/DeepBSDE) with financial-derivative benchmarks, high-dimensional basket options, Monte Carlo control-variate references, robustness experiments, Delta estimation, and direct-\(Z\) supervision diagnostics.

## Project Overview

The Deep BSDE method reformulates a parabolic PDE as a backward stochastic differential equation and approximates its solution using neural networks.

The initial BSDE value \(Y_0\) represents the derivative price. The process \(Z_t\) is related to the sensitivity of the price with respect to the underlying assets.

This repository studies:

- A one-dimensional European call option with an analytic Black–Scholes benchmark
- A 100-dimensional geometric-average basket option with an analytic benchmark
- A 100-dimensional arithmetic-average basket option
- Correlated multi-asset models
- Monte Carlo pricing with a geometric-basket control variate
- Sensitivity to random seeds, time discretization, batch size, and numerical precision
- Delta estimation from both Deep BSDE and common-random-number bump-and-revalue
- Direct supervision of the learned \(Z_t\) process

## Main Contributions

Compared with the original DeepBSDE repository, this project adds:

- Financial PDE classes for Black–Scholes and basket-option problems
- Independent and equicorrelated asset dynamics
- Analytic benchmarks for 1D and geometric-basket options
- A Monte Carlo control-variate estimator for arithmetic-basket options
- Multi-seed error and stability experiments
- Time-discretization, batch-size, and numerical-precision diagnostics
- Dimension scans for 10D, 50D, and 100D problems
- Delta and \(Z_0\) comparison tools
- A preregistered direct-\(Z\) supervision experiment
- Reproducible experiment configurations and summary reports

## Selected Results

### Analytic baselines

| Problem | Dimension | Deep BSDE \(Y_0\) | Reference | Relative error |
|---|---:|---:|---:|---:|
| Black–Scholes European call | 1 | 10.440000 | 10.450584 | 0.1013% |
| Geometric basket call | 100 | 2.976460 | 2.971854 | 0.1550% |

### Arithmetic-basket pricing

A geometric-basket control variate was used to construct high-precision reference values for the 100-dimensional arithmetic-average call.

| Correlation \(\rho\) | MC + control-variate reference | Mean Deep BSDE \(Y_0\) | Mean relative error |
|---:|---:|---:|---:|
| 0.0 | 4.88142052 | 4.88087667 | 0.0322% |
| 0.3 | 7.18101712 | 7.18291000 | 0.0382% |
| 0.5 | 8.30388045 | 8.30603000 | 0.0342% |

Each Deep BSDE result in this table is summarized across three training seeds.

### Robustness observations

The experiments indicate that:

- Seed-to-seed and optimization variability are visible at the sub-percent level.
- Increasing the number of time intervals substantially increases runtime but did not produce a monotonic accuracy improvement over the tested range.
- Float32 reduced runtime in the paired experiments without showing a systematic loss of pricing accuracy.
- Batch size strongly affected convergence speed and stability.
- The correlated geometric-basket model remained within 0.372% of its analytic value across all tested seeds and correlation settings.

### Direct-\(Z\) supervision

The direct-\(Z\) experiment improved Delta and replication performance in the 1D Black–Scholes test. However, the selected model did not pass every preregistered final-gate criterion, so the proposed 100D extension was not performed.

This negative result is retained to document the full experimental decision process and avoid overstating the method's performance.

## Repository Structure

```text
.
├── main.py                     # Main training entry point
├── solver.py                   # TensorFlow/Keras Deep BSDE solver
├── equation.py                 # PDE and financial-equation definitions
├── environment.yml             # Conda environment
├── configs/                    # Reproducible experiment configurations
├── scripts/                    # Experiment and analysis scripts
├── reports/                    # Selected summaries, tables, and figures
├── LICENSE
└── README.md
```

Important scripts include:

| Script | Purpose |
|---|---|
| `compare_bs_1d.py` | Compare the learned 1D price with Black–Scholes |
| `compare_geometric_basket_100d.py` | Compare the 100D geometric-basket result with its analytic value |
| `phase3_mc_cv.py` | Monte Carlo and geometric control-variate engine |
| `run_phase2_error_analysis.py` | Seed, time-grid, precision, batch-size, and correlation experiments |
| `run_phase3_block1.py` | Generate arithmetic-basket reference values |
| `run_phase3_block2_arithmetic.py` | Train the 100D arithmetic-basket Deep BSDE models |
| `run_phase3_block3_formal.py` | Run the formal dimension scan |
| `run_phase3_block4.py` | Compare Deep BSDE and bump-and-revalue Deltas |
| `run_phase3_direct_z_supervision.py` | Run the direct-\(Z\) supervision protocol |

## Installation

The project uses Python 3.11, TensorFlow 2, and Keras 3.

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate deepbsde
```

Some experiment and plotting scripts also require Pillow:

```bash
pip install pillow
```

## Running the Baselines

Run the one-dimensional Black–Scholes experiment:

```bash
python main.py \
  --config_path=configs/bs_1d.json \
  --exp_name=bs_1d
```

Run the 100-dimensional geometric-basket experiment:

```bash
python main.py \
  --config_path=configs/geometric_basket_100d.json \
  --exp_name=geometric_basket_100d
```

Run the 100-dimensional arithmetic-basket experiment:

```bash
python main.py \
  --config_path=configs/arithmetic_basket_100d.json \
  --exp_name=arithmetic_basket_100d
```

Training histories and configurations are written to the `logs/` directory.

## Running the Arithmetic-Basket Study

The arithmetic-basket study is organized as a sequence.

First, generate the Monte Carlo control-variate reference values:

```bash
python scripts/run_phase3_block1.py
```

Then run the three-seed Deep BSDE experiment:

```bash
python scripts/run_phase3_block2_arithmetic.py
```

Run the formal dimension scan:

```bash
python scripts/run_phase3_block3_formal.py
```

Run the Delta comparison:

```bash
python scripts/run_phase3_block4.py
```

These experiments can require significant computation. The default settings reproduce the research protocol rather than providing a lightweight demonstration.

## Direct-\(Z\) Experiment

Run the implementation checks and baseline canary:

```bash
python scripts/run_phase3_direct_z_supervision.py preflight
```

Display the available options:

```bash
python scripts/run_phase3_direct_z_supervision.py --help
```

The final decision and gate results are documented in the corresponding report rather than being summarized only by the best-performing run.

## Methodological Notes

- Reported multi-seed comparisons are descriptive; most experimental conditions use three independent seeds.
- Monte Carlo control-variate timing is not presented as an unconditional speed comparison with Deep BSDE because the two methods serve different purposes.
- Time-discretization conclusions apply only to the tested grids.
- The correlation experiments use a common pairwise correlation across all asset pairs.
- Failed validation gates and negative results are retained rather than removed from the research record.

## References

1. Han, J., Jentzen, A., and E, W.  
   “Solving high-dimensional partial differential equations using deep learning.”  
   *Proceedings of the National Academy of Sciences*, 115(34), 8505–8510, 2018.  
   [https://doi.org/10.1073/pnas.1718942115](https://doi.org/10.1073/pnas.1718942115)

2. E, W., Han, J., and Jentzen, A.  
   “Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations.”  
   *Communications in Mathematics and Statistics*, 5, 349–380, 2017.  
   [https://doi.org/10.1007/s40304-017-0117-6](https://doi.org/10.1007/s40304-017-0117-6)

## Acknowledgements

This repository is based on and extends the TensorFlow implementation of the Deep BSDE solver developed by Jiequn Han and contributors:

[https://github.com/frankhan91/DeepBSDE](https://github.com/frankhan91/DeepBSDE)

The original license and attribution are retained. All financial examples, robustness studies, Monte Carlo benchmarks, Delta diagnostics, and direct-\(Z\) experiments in this repository were developed as extensions to that implementation.

## License

This project retains the license of the original DeepBSDE implementation. See [`LICENSE`](LICENSE) for details.
