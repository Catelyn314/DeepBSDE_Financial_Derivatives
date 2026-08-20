"""Run Phase 3 Block 1: arithmetic-basket ground truth via MC + CV."""

import argparse
import csv
import json
import logging
from pathlib import Path

from phase3_mc_cv import BasketParameters, monte_carlo_with_geometric_control


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "phase3" / "block1_arithmetic_ground_truth"
RHOS = (0.0, 0.3, 0.5)
PATH_SCHEDULE = (1_000_000, 5_000_000, 10_000_000, 20_000_000)
PRECISION_THRESHOLD_PCT = 0.05
MC_SEED = 5101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "block1_run.log", mode="w"),
        ],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], params: BasketParameters) -> None:
    lines = [
        "# Phase 3 Block 1 — Arithmetic-basket ground truth",
        "",
        (
            f"Parameters: S0={params.spot:g}, K={params.strike:g}, "
            f"r={params.rate:g}, sigma={params.sigma:g}, T={params.maturity:g}, "
            f"d={params.dim}, dtype=float64, seed={MC_SEED}."
        ),
        "",
        (
            "Target: discounted arithmetic-average call payoff. Control: "
            "discounted geometric-average call payoff with its analytic expectation."
        ),
        "",
        "| rho | Y0_MC+CV | 95% CI | relative half-width | M | time (s) | status |",
        "|---:|---:|:---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rho']:.1f} | {row['y0_mc_cv']:.10f} | "
            f"[{row['ci95_low']:.10f}, {row['ci95_high']:.10f}] | "
            f"{row['relative_ci_half_width_pct']:.8f}% | "
            f"{row['paths']:,} | {row['elapsed_seconds']:.3f} | "
            f"{'pass' if row['precision_reached'] else 'hit cap'} |"
        )
    lines.extend(
        [
            "",
            "## Control-variate diagnostics",
            "",
            "| rho | beta | raw MC SE | CV SE | variance-reduction factor | geometric control price |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['rho']:.1f} | {row['beta']:.10f} | "
            f"{row['raw_mc_standard_error']:.10f} | "
            f"{row['cv_standard_error']:.10f} | "
            f"{row['variance_reduction_factor']:.4f} | "
            f"{row['analytic_geometric_control_price']:.10f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir)
    params = BasketParameters()
    rows = []

    for rho in RHOS:
        selected = None
        for paths in PATH_SCHEDULE:
            logging.info(
                "Starting rho=%.1f with M=%d, target=arithmetic, control=geometric",
                rho,
                paths,
            )
            result = monte_carlo_with_geometric_control(
                params=params,
                rho=rho,
                paths=paths,
                seed=MC_SEED,
                target="arithmetic",
                chunk_size=args.chunk_size,
            )
            precise = (
                result.relative_ci_half_width_pct <= PRECISION_THRESHOLD_PCT
            )
            if result.cv_standard_error > 0.0:
                variance_reduction = (
                    result.raw_standard_error / result.cv_standard_error
                ) ** 2
            else:
                variance_reduction = float("inf")
            selected = {
                "rho": result.rho,
                "paths": result.paths,
                "seed": result.seed,
                "y0_mc_cv": result.cv_price,
                "ci95_low": result.ci95_low,
                "ci95_high": result.ci95_high,
                "ci95_half_width": result.ci95_half_width,
                "relative_ci_half_width_pct": result.relative_ci_half_width_pct,
                "precision_threshold_pct": PRECISION_THRESHOLD_PCT,
                "precision_reached": precise,
                "hit_path_cap": paths == PATH_SCHEDULE[-1] and not precise,
                "elapsed_seconds": result.elapsed_seconds,
                "raw_mc_price": result.raw_price,
                "raw_mc_standard_error": result.raw_standard_error,
                "cv_standard_error": result.cv_standard_error,
                "beta": result.beta,
                "control_variance": result.control_variance,
                "variance_reduction_factor": variance_reduction,
                "analytic_geometric_control_price": result.analytic_control_price,
            }
            logging.info(
                (
                    "Completed rho=%.1f: Y0_MC+CV=%.10f, 95%% CI="
                    "[%.10f, %.10f], rel_half_width=%.8f%%, beta=%.10f, "
                    "VRF=%.4f, elapsed=%.3fs"
                ),
                rho,
                result.cv_price,
                result.ci95_low,
                result.ci95_high,
                result.relative_ci_half_width_pct,
                result.beta,
                variance_reduction,
                result.elapsed_seconds,
            )
            if precise:
                break
            if paths < PATH_SCHEDULE[-1]:
                logging.info(
                    "rho=%.1f missed <=%.2f%%; increasing path count.",
                    rho,
                    PRECISION_THRESHOLD_PCT,
                )
        rows.append(selected)
        logging.info(
            "Subtask summary rho=%.1f: %s at M=%d.",
            rho,
            "PASS" if selected["precision_reached"] else "HIT CAP",
            selected["paths"],
        )

    write_csv(args.output_dir / "block1_ground_truth.csv", rows)
    (args.output_dir / "block1_ground_truth.json").write_text(
        json.dumps(
            {
                "parameters": params.__dict__,
                "mc_seed": MC_SEED,
                "path_schedule": PATH_SCHEDULE,
                "precision_threshold_pct": PRECISION_THRESHOLD_PCT,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    write_summary(args.output_dir / "block1_summary.md", rows, params)
    passed = sum(row["precision_reached"] for row in rows)
    logging.info(
        "Block 1 complete: %d/%d rho cases reached precision. "
        "Stopping before Block 2.",
        passed,
        len(rows),
    )
    if passed != len(rows):
        raise SystemExit("Block 1 precision target not reached; do not continue.")


if __name__ == "__main__":
    main()
