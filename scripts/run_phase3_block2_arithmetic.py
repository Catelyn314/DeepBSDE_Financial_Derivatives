"""Run Phase 3 v2 Block 2 arithmetic-basket Deep BSDE experiments."""

import csv
import json
from pathlib import Path

from run_phase2_error_analysis import (
    BASE_BASKET_CONFIG,
    align_histories,
    draw_mean_band_chart,
    load_json,
    make_config,
    run_training,
    settling_step,
    stats,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "phase3" / "block2_arithmetic_bsde"
RUN_DIR = OUTPUT_DIR / "runs"
GROUND_TRUTH_PATH = (
    ROOT
    / "reports"
    / "phase3"
    / "block1_arithmetic_ground_truth"
    / "block1_ground_truth.json"
)
SEEDS = (5101, 5201, 5202)
RHOS = (0.0, 0.3, 0.5)
ERROR_BAND_PCT = 0.5
TRIGGER_ERROR_PCT = 0.35
TRIGGER_SETTLING_STEP = 4000.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truths() -> dict[float, float]:
    with GROUND_TRUTH_PATH.open() as source:
        payload = json.load(source)
    values = {
        float(row["rho"]): float(row["y0_mc_cv"])
        for row in payload["results"]
    }
    if set(values) != set(RHOS):
        raise RuntimeError("Block 1 ground truth does not contain all required rho values.")
    return values


def build_config(base: dict, rho: float, ground_truth: float) -> dict:
    equation_updates = {
        "eqn_name": "ArithmeticBasket100D",
        "num_time_interval": 25,
        "sample_dtype": "float32",
        "rho": rho,
        "ground_truth": ground_truth,
        "_comment": f"100D arithmetic basket with equicorrelation rho={rho}.",
    }
    # Reuse the Phase 2 correlation protocol: a symmetric width-3 initialization
    # interval around the known reference, applied consistently to every rho.
    return make_config(
        base,
        eqn_updates=equation_updates,
        net_updates={
            "num_hiddens": [110, 110],
            "num_iterations": 6000,
            "batch_size": 128,
            "logging_frequency": 25,
            "dtype": "float32",
            "y_init_range": [
                max(0.0, ground_truth - 3.0),
                ground_truth + 3.0,
            ],
        },
    )


def write_summary(path: Path, rows: list[dict], summaries: list[dict]) -> None:
    lines = [
        "# Phase 3 Block 2 — Arithmetic-basket Deep BSDE",
        "",
        (
            "100D; N=25; batch=128; network=[110,110]; max_iter=6000; "
            "training dtype=float32; seeds=5101,5201,5202."
        ),
        "",
        "## Per-seed results",
        "",
        "| rho | seed | ground truth | Y0 | relative error | settling step | time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        settling = (
            str(row["settling_step"])
            if row["settling_step"] is not None
            else "not settled"
        )
        lines.append(
            f"| {row['rho']:.1f} | {row['seed']} | "
            f"{row['ground_truth']:.10f} | {row['y0']:.10f} | "
            f"{row['relative_error_pct']:.8f}% | {settling} | "
            f"{row['elapsed_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed summaries",
            "",
            "| rho | Y0 mean ± SD | relative error mean ± SD | settling step mean ± SD | trigger rule |",
            "|---:|:---|:---|:---|:---:|",
        ]
    )
    for row in summaries:
        settling = (
            f"{row['settling_step_mean']:.2f} ± {row['settling_step_sd']:.2f}"
            if row["settling_step_mean"] is not None
            else "not available"
        )
        lines.append(
            f"| {row['rho']:.1f} | "
            f"{row['y0_mean']:.10f} ± {row['y0_sd']:.10f} | "
            f"{row['relative_error_mean_pct']:.8f}% ± "
            f"{row['relative_error_sd_pct']:.8f}% | "
            f"{settling} | "
            f"{'trigger' if row['trigger_rule_met'] else 'do not trigger'} |"
        )
    lines.extend(
        [
            "",
            (
                "Fixed trigger rule: mean relative error >0.35% AND mean "
                "settling step >4000."
            ),
            "",
            "Block 2.5 was not started automatically.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_json(BASE_BASKET_CONFIG)
    ground_truths = load_ground_truths()
    raw_rows = []
    summaries = []

    for rho in RHOS:
        ground_truth = ground_truths[rho]
        config = build_config(base, rho, ground_truth)
        histories = []
        rho_rows = []
        for seed in SEEDS:
            rho_label = str(rho).replace(".", "p")
            run_name = f"arithmetic100d_rho_{rho_label}_float32_seed_{seed}"
            print(f"Starting rho={rho:.1f}, seed={seed}", flush=True)
            history, _ = run_training(config, RUN_DIR, run_name, seed)
            histories.append(history)
            final = history[-1]
            relative_error = abs(final["y0"] - ground_truth) / ground_truth * 100.0
            settled = settling_step(history, ground_truth, ERROR_BAND_PCT)
            settled_value = int(settled) if settled != "" else None
            row = {
                "rho": rho,
                "seed": seed,
                "ground_truth": ground_truth,
                "y0": final["y0"],
                "absolute_error": abs(final["y0"] - ground_truth),
                "relative_error_pct": relative_error,
                "settling_step": settled_value,
                "final_loss": final["loss"],
                "elapsed_seconds": final["elapsed"],
            }
            raw_rows.append(row)
            rho_rows.append(row)
            print(
                f"Subtask summary rho={rho:.1f}, seed={seed}: "
                f"Y0={final['y0']:.10f}, error={relative_error:.8f}%, "
                f"settling={settled_value}, elapsed={final['elapsed']:.1f}s",
                flush=True,
            )

        y0_mean, y0_sd = stats([row["y0"] for row in rho_rows])
        error_mean, error_sd = stats(
            [row["relative_error_pct"] for row in rho_rows]
        )
        settling_values = [
            row["settling_step"]
            for row in rho_rows
            if row["settling_step"] is not None
        ]
        if len(settling_values) == len(rho_rows):
            settling_mean, settling_sd = stats(settling_values)
            trigger = (
                error_mean > TRIGGER_ERROR_PCT
                and settling_mean > TRIGGER_SETTLING_STEP
            )
        else:
            settling_mean = None
            settling_sd = None
            trigger = False
        summaries.append(
            {
                "rho": rho,
                "n": len(rho_rows),
                "ground_truth": ground_truth,
                "y0_mean": y0_mean,
                "y0_sd": y0_sd,
                "relative_error_mean_pct": error_mean,
                "relative_error_sd_pct": error_sd,
                "settling_step_mean": settling_mean,
                "settling_step_sd": settling_sd,
                "num_not_settled": len(rho_rows) - len(settling_values),
                "trigger_error_threshold_pct": TRIGGER_ERROR_PCT,
                "trigger_settling_threshold": TRIGGER_SETTLING_STEP,
                "trigger_rule_met": trigger,
            }
        )
        draw_mean_band_chart(
            [
                {
                    "label": f"rho={rho:.1f} mean ± SD",
                    "points": align_histories(histories),
                }
            ],
            f"Arithmetic Basket Deep BSDE — rho={rho:.1f}",
            "100D; float32; three seeds; mean ± one sample SD",
            OUTPUT_DIR / f"rho_{str(rho).replace('.', 'p')}_y0_convergence.png",
            analytic=ground_truth,
            reference_label="MC+CV ground truth",
        )
        print(
            f"rho={rho:.1f} group complete: error={error_mean:.8f}% ± "
            f"{error_sd:.8f}%, settling={settling_mean} ± {settling_sd}",
            flush=True,
        )

    write_csv(
        OUTPUT_DIR / "block2_arithmetic_raw.csv",
        raw_rows,
        [
            "rho",
            "seed",
            "ground_truth",
            "y0",
            "absolute_error",
            "relative_error_pct",
            "settling_step",
            "final_loss",
            "elapsed_seconds",
        ],
    )
    write_csv(
        OUTPUT_DIR / "block2_arithmetic_summary.csv",
        summaries,
        [
            "rho",
            "n",
            "ground_truth",
            "y0_mean",
            "y0_sd",
            "relative_error_mean_pct",
            "relative_error_sd_pct",
            "settling_step_mean",
            "settling_step_sd",
            "num_not_settled",
            "trigger_error_threshold_pct",
            "trigger_settling_threshold",
            "trigger_rule_met",
        ],
    )
    (OUTPUT_DIR / "block2_arithmetic_results.json").write_text(
        json.dumps(
            {
                "fixed_config": {
                    "dim": 100,
                    "num_time_interval": 25,
                    "batch_size": 128,
                    "num_hiddens": [110, 110],
                    "num_iterations": 6000,
                    "training_dtype": "float32",
                    "logging_frequency": 25,
                    "seeds": SEEDS,
                },
                "ground_truth_source": str(GROUND_TRUTH_PATH),
                "runs": raw_rows,
                "summaries": summaries,
                "importance_sampling_triggered_for_any_rho": any(
                    row["trigger_rule_met"] for row in summaries
                ),
            },
            indent=2,
        )
        + "\n"
    )
    write_summary(OUTPUT_DIR / "block2_arithmetic_summary.md", raw_rows, summaries)
    print(
        "Block 2 formal experiments complete. Stopping before Block 2.5.",
        flush=True,
    )


if __name__ == "__main__":
    main()
