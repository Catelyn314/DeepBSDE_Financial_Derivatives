"""Run Phase 3 Block 2: Deep BSDE on the arithmetic-average basket."""

import argparse
import copy
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import equation as eqn
from solver import BSDESolver


BASE_CONFIG = ROOT / "configs" / "arithmetic_basket_100d.json"
GROUND_TRUTH_FILE = (
    ROOT
    / "reports"
    / "phase3"
    / "block1_arithmetic_ground_truth"
    / "block1_ground_truth.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "phase3" / "block2_deep_bsde"
SEEDS = (5101, 5201, 5202)
RHOS = (0.0, 0.3, 0.5)
SETTLING_THRESHOLD_PCT = 0.5
TRIGGER_ERROR_PCT = 0.35
TRIGGER_SETTLING_STEP = 4000


class DictToObject:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)


class Config:
    def __init__(self, config_dict):
        self.eqn_config = DictToObject(config_dict["eqn_config"])
        self.net_config = DictToObject(config_dict["net_config"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "block2_run.log", mode="a"),
        ],
    )


def load_json(path: Path) -> dict:
    with path.open() as input_file:
        return json.load(input_file)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_history(path: Path) -> list[dict]:
    with path.open(newline="") as input_file:
        return [
            {
                "step": int(row["step"]),
                "loss": float(row["loss_function"]),
                "y0": float(row["target_value"]),
                "elapsed": float(row["elapsed_time"]),
            }
            for row in csv.DictReader(input_file)
        ]


def save_history(path: Path, history: np.ndarray) -> None:
    np.savetxt(
        path,
        history,
        fmt=["%d", "%.10e", "%.10e", "%.6f"],
        delimiter=",",
        header="step,loss_function,target_value,elapsed_time",
        comments="",
    )


def relative_error_pct(value: float, reference: float) -> float:
    return abs(value - reference) / reference * 100.0


def settling_step(history: list[dict], reference: float) -> Optional[int]:
    outside = [
        index
        for index, row in enumerate(history)
        if relative_error_pct(row["y0"], reference) > SETTLING_THRESHOLD_PCT
    ]
    if not outside:
        return history[0]["step"] if history else None
    last_outside = outside[-1]
    if last_outside == len(history) - 1:
        return None
    return history[last_outside + 1]["step"]


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1))


def load_ground_truths() -> dict[float, float]:
    payload = load_json(GROUND_TRUTH_FILE)
    values = {
        float(row["rho"]): float(row["y0_mc_cv"])
        for row in payload["results"]
    }
    if set(values) != set(RHOS):
        raise ValueError(f"Expected ground truths for {RHOS}, got {values}.")
    return values


def make_config(base: dict, rho: float, ground_truth: float) -> dict:
    config = copy.deepcopy(base)
    config["eqn_config"].update(
        {
            "rho": rho,
            "ground_truth": ground_truth,
            "_comment": (
                "Phase 3 Block 2 arithmetic basket, "
                f"equicorrelation rho={rho}."
            ),
        }
    )
    config["net_config"].update(
        {
            "y_init_range": [
                max(0.0, ground_truth - 3.0),
                ground_truth + 3.0,
            ],
            "num_hiddens": [110, 110],
            "num_iterations": 6000,
            "batch_size": 128,
            "logging_frequency": 25,
            "dtype": "float64",
        }
    )
    return config


def run_one(
    config_dict: dict,
    run_dir: Path,
    run_name: str,
    seed: int,
) -> list[dict]:
    history_path = run_dir / f"{run_name}_training_history.csv"
    if history_path.exists():
        logging.info("Reusing completed run %s", run_name)
        return read_history(history_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / f"{run_name}_config.json"
    with config_path.open("w") as output:
        json.dump(config_dict, output, indent=2)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.set_floatx("float64")

    config = Config(config_dict)
    basket = eqn.ArithmeticBasket100D(config.eqn_config)
    solver = BSDESolver(config, basket)
    logging.info("Training %s", run_name)
    history_array = solver.train()
    save_history(history_path, history_array)

    z0 = solver.model.z_init.numpy().reshape(-1)
    np.save(run_dir / f"{run_name}_z0.npy", z0)
    np.savetxt(
        run_dir / f"{run_name}_z0.csv",
        z0,
        delimiter=",",
        header="z0",
        comments="",
    )
    logging.info(
        "Finished %s: Y0=%.10f, loss=%.6e, elapsed=%.1fs",
        run_name,
        history_array[-1, 2],
        history_array[-1, 1],
        history_array[-1, 3],
    )
    return read_history(history_path)


def build_summary(raw_rows: list[dict]) -> list[dict]:
    summary = []
    for rho in RHOS:
        items = [row for row in raw_rows if row["rho"] == rho]
        y0_values = [row["y0"] for row in items]
        error_values = [row["relative_error_pct"] for row in items]
        settling_values = [
            row["settling_step"]
            for row in items
            if row["settling_step"] is not None
        ]
        summary.append(
            {
                "rho": rho,
                "num_runs": len(items),
                "ground_truth": items[0]["ground_truth"],
                "y0_mean": float(np.mean(y0_values)),
                "y0_sd": sample_sd(y0_values),
                "relative_error_pct_mean": float(np.mean(error_values)),
                "relative_error_pct_sd": sample_sd(error_values),
                "settling_step_mean": (
                    float(np.mean(settling_values))
                    if len(settling_values) == len(items)
                    else None
                ),
                "settling_step_sd": (
                    sample_sd(settling_values)
                    if len(settling_values) == len(items)
                    else None
                ),
                "num_not_settled": len(items) - len(settling_values),
                "elapsed_seconds_mean": float(
                    np.mean([row["elapsed_seconds"] for row in items])
                ),
                "elapsed_seconds_sd": sample_sd(
                    [row["elapsed_seconds"] for row in items]
                ),
            }
        )
    return summary


def trigger_decision(summary: list[dict]) -> dict:
    per_rho = []
    any_triggered = False
    any_boundary = False
    for row in summary:
        mean_error = row["relative_error_pct_mean"]
        mean_settling = row["settling_step_mean"]
        triggered = (
            mean_settling is not None
            and mean_error > TRIGGER_ERROR_PCT
            and mean_settling > TRIGGER_SETTLING_STEP
        )
        boundary = 0.33 <= mean_error <= 0.37
        unsettled = row["num_not_settled"] > 0
        per_rho.append(
            {
                "rho": row["rho"],
                "mean_relative_error_pct": mean_error,
                "mean_settling_step": mean_settling,
                "num_not_settled": row["num_not_settled"],
                "triggered": triggered,
                "boundary_error_case": boundary,
                "decision_requires_review_due_to_unsettled": unsettled,
            }
        )
        any_triggered = any_triggered or triggered
        any_boundary = any_boundary or boundary
    return {
        "rule": (
            "mean relative error > 0.35% AND mean settling step > 4000"
        ),
        "per_rho": per_rho,
        "block_2_5_triggered": any_triggered,
        "has_boundary_case": any_boundary,
    }


def write_summary_markdown(
    path: Path,
    raw_rows: list[dict],
    summary: list[dict],
    decision: dict,
) -> None:
    lines = [
        "# Phase 3 Block 2 — Deep BSDE arithmetic basket",
        "",
        "Fixed settings: d=100, N=25, batch=128, max_iter=6000, "
        "dtype=float64, hidden layers=[110,110], logging every 25 steps.",
        "",
        "## Per-seed results",
        "",
        "| rho | seed | Y0 | relative error | settling step | time (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in raw_rows:
        settling = (
            str(row["settling_step"])
            if row["settling_step"] is not None
            else "not settled"
        )
        lines.append(
            f"| {row['rho']:.1f} | {row['seed']} | {row['y0']:.10f} | "
            f"{row['relative_error_pct']:.6f}% | {settling} | "
            f"{row['elapsed_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed mean ± sample SD",
            "",
            "| rho | Y0 | relative error | settling step | not settled |",
            "|---:|:---|:---|:---|---:|",
        ]
    )
    for row in summary:
        settling = (
            f"{row['settling_step_mean']:.1f} ± "
            f"{row['settling_step_sd']:.1f}"
            if row["settling_step_mean"] is not None
            else "not estimable"
        )
        lines.append(
            f"| {row['rho']:.1f} | {row['y0_mean']:.10f} ± "
            f"{row['y0_sd']:.10f} | "
            f"{row['relative_error_pct_mean']:.6f}% ± "
            f"{row['relative_error_pct_sd']:.6f}% | {settling} | "
            f"{row['num_not_settled']} |"
        )
    lines.extend(["", "## Trigger decision", ""])
    for row in decision["per_rho"]:
        settling = (
            f"{row['mean_settling_step']:.1f}"
            if row["mean_settling_step"] is not None
            else "not estimable"
        )
        lines.append(
            f"- rho={row['rho']:.1f}: mean error "
            f"{row['mean_relative_error_pct']:.6f}%, mean settling "
            f"{settling}; triggered={row['triggered']}."
        )
    lines.append(
        f"- Overall Block 2.5 trigger: {decision['block_2_5_triggered']}."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir)
    base = load_json(BASE_CONFIG)
    ground_truths = load_ground_truths()
    run_dir = args.output_dir / "runs"
    raw_rows = []

    for rho in RHOS:
        ground_truth = ground_truths[rho]
        for seed_id, seed in enumerate(SEEDS):
            config = make_config(base, rho, ground_truth)
            rho_label = str(rho).replace(".", "p")
            run_name = f"arithmetic100d_rho_{rho_label}_seed_{seed}"
            history = run_one(config, run_dir, run_name, seed)
            final = history[-1]
            settled = settling_step(history, ground_truth)
            row = {
                "rho": rho,
                "seed_id": seed_id,
                "seed": seed,
                "ground_truth": ground_truth,
                "y0": final["y0"],
                "absolute_error": abs(final["y0"] - ground_truth),
                "relative_error_pct": relative_error_pct(
                    final["y0"], ground_truth
                ),
                "settling_step": settled,
                "final_loss": final["loss"],
                "elapsed_seconds": final["elapsed"],
            }
            raw_rows.append(row)
            logging.info(
                "Subtask summary rho=%.1f seed=%d: error=%.6f%%, settling=%s",
                rho,
                seed,
                row["relative_error_pct"],
                settled if settled is not None else "not settled",
            )

    raw_fields = [
        "rho",
        "seed_id",
        "seed",
        "ground_truth",
        "y0",
        "absolute_error",
        "relative_error_pct",
        "settling_step",
        "final_loss",
        "elapsed_seconds",
    ]
    write_csv(args.output_dir / "block2_per_seed.csv", raw_rows, raw_fields)
    summary = build_summary(raw_rows)
    summary_fields = list(summary[0])
    write_csv(args.output_dir / "block2_summary.csv", summary, summary_fields)
    decision = trigger_decision(summary)
    with (args.output_dir / "block2_results.json").open("w") as output:
        json.dump(
            {
                "seeds": SEEDS,
                "fixed_settings": {
                    "dim": 100,
                    "num_time_interval": 25,
                    "batch_size": 128,
                    "num_iterations": 6000,
                    "dtype": "float64",
                    "num_hiddens": [110, 110],
                    "settling_threshold_pct": SETTLING_THRESHOLD_PCT,
                },
                "per_seed": raw_rows,
                "summary": summary,
                "trigger_decision": decision,
            },
            output,
            indent=2,
        )
        output.write("\n")
    write_summary_markdown(
        args.output_dir / "block2_summary.md",
        raw_rows,
        summary,
        decision,
    )
    logging.info(
        "Block 2 complete. Block 2.5 triggered=%s. Stopping.",
        decision["block_2_5_triggered"],
    )


if __name__ == "__main__":
    main()
