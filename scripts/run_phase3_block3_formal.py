"""Run Phase 3 v2 Block 3 formal time-to-accuracy dimension scan."""

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from phase3_mc_cv import BasketParameters, monte_carlo_with_geometric_control
from run_phase2_error_analysis import (
    AMBER,
    BLUE,
    GREEN,
    GRID,
    INK,
    MUTED,
    PURPLE,
    BASE_BASKET_CONFIG,
    Config,
    font,
    load_json,
    make_config,
    read_history,
    run_training,
    settling_step,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "phase3" / "block3_formal"
RUN_DIR = OUTPUT_DIR / "runs"
BLOCK1_PATH = (
    ROOT
    / "reports"
    / "phase3"
    / "block1_arithmetic_ground_truth"
    / "block1_ground_truth.json"
)
D100_HISTORY = (
    ROOT
    / "reports"
    / "phase3"
    / "block2_arithmetic_bsde"
    / "runs"
    / "arithmetic100d_rho_0p0_float32_seed_5101_training_history.csv"
)
DIMS = (10, 50, 100)
TARGETS_PCT = (0.5, 0.2)
GROUND_TRUTH_PATHS = 1_000_000
GROUND_TRUTH_SEED = 5101
BSDE_SEED = 5101
MC_SCAN_SEED = 5201
MC_PATH_SCHEDULE = (
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_d100_ground_truth() -> dict:
    with BLOCK1_PATH.open() as source:
        payload = json.load(source)
    return next(
        row for row in payload["results"] if float(row["rho"]) == 0.0
    )


def build_bsde_config(base: dict, dim: int, ground_truth: float) -> dict:
    return make_config(
        base,
        eqn_updates={
            "eqn_name": "ArithmeticBasketIndependent",
            "dim": dim,
            "num_time_interval": 25,
            "sample_dtype": "float32",
            "ground_truth": ground_truth,
            "_comment": f"Block 3 formal: {dim}D arithmetic basket, rho=0.",
        },
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


def first_hit(history: list[dict], reference: float, target_pct: float):
    for row in history:
        error = abs(row["y0"] - reference) / reference * 100.0
        if error < target_pct:
            return row["step"], row["elapsed"], error
    return None, None, None


def elapsed_for_step(history: list[dict], step):
    if step in (None, ""):
        return None
    for row in history:
        if row["step"] >= int(step):
            return row["elapsed"]
    return None


def create_time_plot(rows: list[dict], path: Path) -> None:
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text(
        (100, 28),
        "Block 3 Time-to-Accuracy vs Dimension",
        fill=INK,
        font=font(30, bold=True),
    )
    draw.text(
        (100, 66),
        "First-hit wall-clock time; logarithmic y-axis",
        fill=MUTED,
        font=font(18),
    )
    left, top, right, bottom = 105, 112, 1145, 660
    positive_times = [
        float(row["wall_clock_seconds"])
        for row in rows
        if row["reached_target"] and float(row["wall_clock_seconds"]) > 0.0
    ]
    y_min = 10 ** math.floor(math.log10(min(positive_times)) - 0.25)
    y_max = 10 ** math.ceil(math.log10(max(positive_times)) + 0.25)
    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def px(dim):
        return left + (dim - min(DIMS)) / (max(DIMS) - min(DIMS)) * (right - left)

    def py(value):
        fraction = (math.log10(value) - log_min) / (log_max - log_min)
        return bottom - fraction * (bottom - top)

    exponent = math.floor(log_min)
    while exponent <= math.ceil(log_max):
        tick = 10 ** exponent
        if y_min <= tick <= y_max:
            y = py(tick)
            draw.line((left, y, right, y), fill=GRID, width=1)
            label = f"{tick:g} s"
            bbox = draw.textbbox((0, 0), label, font=font(15))
            draw.text(
                (left - 14 - (bbox[2] - bbox[0]), y - 9),
                label,
                fill=MUTED,
                font=font(15),
            )
        exponent += 1
    for dim in DIMS:
        x = px(dim)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = str(dim)
        bbox = draw.textbbox((0, 0), label, font=font(16))
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2, bottom + 18),
            label,
            fill=MUTED,
            font=font(16),
        )
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    draw.line((left, top, left, bottom), fill=INK, width=2)
    draw.text(
        ((left + right) / 2 - 45, 710),
        "Dimension d",
        fill=INK,
        font=font(18),
    )

    series_specs = [
        ("Deep BSDE", 0.5, BLUE, "Deep BSDE <0.5%", -5),
        ("Deep BSDE", 0.2, GREEN, "Deep BSDE <0.2%", 5),
        ("MC+CV", 0.5, AMBER, "MC+CV <0.5%", -5),
        ("MC+CV", 0.2, PURPLE, "MC+CV <0.2%", 5),
    ]
    for series_index, (method, target, color, label, x_offset) in enumerate(
        series_specs
    ):
        matching = sorted(
            [
                row
                for row in rows
                if row["method"] == method
                and float(row["target_relative_error_pct"]) == target
                and row["reached_target"]
            ],
            key=lambda row: row["dim"],
        )
        points = [
            (
                px(int(row["dim"])) + x_offset,
                py(float(row["wall_clock_seconds"])),
            )
            for row in matching
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for (x, y), row in zip(points, matching):
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
            value = float(row["wall_clock_seconds"])
            value_label = f"{value:.3g}s"
            draw.text(
                (x + 9, y - 22 + series_index % 2 * 18),
                value_label,
                fill=INK,
                font=font(13),
            )
        legend_x = 735 + (series_index % 2) * 205
        legend_y = 28 + (series_index // 2) * 31
        draw.line(
            (legend_x, legend_y + 11, legend_x + 36, legend_y + 11),
            fill=color,
            width=4,
        )
        draw.text(
            (legend_x + 45, legend_y),
            label,
            fill=INK,
            font=font(15),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_json(BASE_BASKET_CONFIG)
    ground_truth_rows = []
    ground_truths = {}

    for dim in DIMS:
        if dim == 100:
            source = load_d100_ground_truth()
            row = {
                "dim": dim,
                "price": float(source["y0_mc_cv"]),
                "ci95_low": float(source["ci95_low"]),
                "ci95_high": float(source["ci95_high"]),
                "relative_ci_half_width_pct": float(
                    source["relative_ci_half_width_pct"]
                ),
                "paths": int(source["paths"]),
                "elapsed_seconds": float(source["elapsed_seconds"]),
                "seed": int(source["seed"]),
                "source": "reused Block 1",
                "precision_reached": bool(source["precision_reached"]),
            }
        else:
            result = monte_carlo_with_geometric_control(
                BasketParameters(dim=dim),
                rho=0.0,
                paths=GROUND_TRUTH_PATHS,
                seed=GROUND_TRUTH_SEED,
                target="arithmetic",
            )
            row = {
                "dim": dim,
                "price": result.cv_price,
                "ci95_low": result.ci95_low,
                "ci95_high": result.ci95_high,
                "relative_ci_half_width_pct": result.relative_ci_half_width_pct,
                "paths": result.paths,
                "elapsed_seconds": result.elapsed_seconds,
                "seed": result.seed,
                "source": "formal Block 3 MC+CV",
                "precision_reached": (
                    result.relative_ci_half_width_pct <= 0.05
                ),
            }
        if not row["precision_reached"]:
            raise RuntimeError(
                f"d={dim} ground truth did not reach <=0.05% precision."
            )
        ground_truth_rows.append(row)
        ground_truths[dim] = row["price"]
        print(
            f"Ground truth d={dim}: {row['price']:.10f}, "
            f"relative CI half-width={row['relative_ci_half_width_pct']:.8f}%, "
            f"time={row['elapsed_seconds']:.4f}s",
            flush=True,
        )

    write_csv(
        OUTPUT_DIR / "block3_ground_truths.csv",
        ground_truth_rows,
        [
            "dim",
            "price",
            "ci95_low",
            "ci95_high",
            "relative_ci_half_width_pct",
            "paths",
            "elapsed_seconds",
            "seed",
            "source",
            "precision_reached",
        ],
    )

    time_rows = []
    settling_rows = []
    for dim in DIMS:
        ground_truth = ground_truths[dim]
        if dim == 100:
            history = read_history(D100_HISTORY)
            history_source = "reused Block 2"
        else:
            config = build_bsde_config(base, dim, ground_truth)
            run_name = (
                f"arithmetic_d{dim}_rho0_float32_seed_{BSDE_SEED}_6000step"
            )
            print(f"Starting formal Deep BSDE d={dim}", flush=True)
            history, _ = run_training(
                config,
                RUN_DIR,
                run_name,
                BSDE_SEED,
            )
            history_source = "formal Block 3"

        for target_pct in TARGETS_PCT:
            step, elapsed, error = first_hit(
                history,
                ground_truth,
                target_pct,
            )
            time_rows.append(
                {
                    "dim": dim,
                    "method": "Deep BSDE",
                    "target_relative_error_pct": target_pct,
                    "reached_target": step is not None,
                    "wall_clock_seconds": elapsed,
                    "step_or_paths": step,
                    "actual_relative_error_pct": error,
                    "hit_path_cap": False,
                    "source": history_source,
                }
            )
            settled = settling_step(history, ground_truth, target_pct)
            settled_value = int(settled) if settled != "" else None
            settling_rows.append(
                {
                    "dim": dim,
                    "target_relative_error_pct": target_pct,
                    "settling_step": settled_value,
                    "settling_wall_clock_seconds": elapsed_for_step(
                        history,
                        settled_value,
                    ),
                    "first_hit_step": step,
                    "first_hit_wall_clock_seconds": elapsed,
                }
            )
        print(
            f"Deep BSDE d={dim} complete: final step={history[-1]['step']}, "
            f"time={history[-1]['elapsed']:.1f}s",
            flush=True,
        )

    mc_checkpoint_rows = []
    for dim in DIMS:
        ground_truth = ground_truths[dim]
        unresolved = set(TARGETS_PCT)
        for paths in MC_PATH_SCHEDULE:
            result = monte_carlo_with_geometric_control(
                BasketParameters(dim=dim),
                rho=0.0,
                paths=paths,
                seed=MC_SCAN_SEED,
                target="arithmetic",
            )
            relative_error = (
                abs(result.cv_price - ground_truth) / ground_truth * 100.0
            )
            mc_checkpoint_rows.append(
                {
                    "dim": dim,
                    "paths": paths,
                    "estimate": result.cv_price,
                    "ground_truth": ground_truth,
                    "relative_error_pct": relative_error,
                    "ci95_half_width": result.ci95_half_width,
                    "relative_ci_half_width_pct": (
                        result.ci95_half_width / result.cv_price * 100.0
                    ),
                    "elapsed_seconds": result.elapsed_seconds,
                    "seed": MC_SCAN_SEED,
                }
            )
            for target_pct in list(unresolved):
                if relative_error < target_pct:
                    time_rows.append(
                        {
                            "dim": dim,
                            "method": "MC+CV",
                            "target_relative_error_pct": target_pct,
                            "reached_target": True,
                            "wall_clock_seconds": result.elapsed_seconds,
                            "step_or_paths": paths,
                            "actual_relative_error_pct": relative_error,
                            "hit_path_cap": False,
                            "source": "formal Block 3 independent scan",
                        }
                    )
                    unresolved.remove(target_pct)
            print(
                f"MC+CV d={dim}, M={paths}: error={relative_error:.8f}%, "
                f"time={result.elapsed_seconds:.6f}s",
                flush=True,
            )
            if not unresolved:
                break
        if unresolved:
            final = mc_checkpoint_rows[-1]
            for target_pct in unresolved:
                time_rows.append(
                    {
                        "dim": dim,
                        "method": "MC+CV",
                        "target_relative_error_pct": target_pct,
                        "reached_target": False,
                        "wall_clock_seconds": final["elapsed_seconds"],
                        "step_or_paths": final["paths"],
                        "actual_relative_error_pct": final[
                            "relative_error_pct"
                        ],
                        "hit_path_cap": True,
                        "source": "formal Block 3 independent scan",
                    }
                )

    time_rows.sort(
        key=lambda row: (
            row["dim"],
            row["method"],
            -row["target_relative_error_pct"],
        )
    )
    write_csv(
        OUTPUT_DIR / "block3_time_to_accuracy.csv",
        time_rows,
        [
            "dim",
            "method",
            "target_relative_error_pct",
            "reached_target",
            "wall_clock_seconds",
            "step_or_paths",
            "actual_relative_error_pct",
            "hit_path_cap",
            "source",
        ],
    )
    write_csv(
        OUTPUT_DIR / "block3_mc_checkpoints.csv",
        mc_checkpoint_rows,
        [
            "dim",
            "paths",
            "estimate",
            "ground_truth",
            "relative_error_pct",
            "ci95_half_width",
            "relative_ci_half_width_pct",
            "elapsed_seconds",
            "seed",
        ],
    )
    write_csv(
        OUTPUT_DIR / "block3_bsde_stability_diagnostic.csv",
        settling_rows,
        [
            "dim",
            "target_relative_error_pct",
            "settling_step",
            "settling_wall_clock_seconds",
            "first_hit_step",
            "first_hit_wall_clock_seconds",
        ],
    )
    (OUTPUT_DIR / "block3_results.json").write_text(
        json.dumps(
            {
                "status": "formal Block 3 complete",
                "settings": {
                    "dims": DIMS,
                    "rho": 0.0,
                    "targets_pct": TARGETS_PCT,
                    "bsde_seed": BSDE_SEED,
                    "mc_scan_seed": MC_SCAN_SEED,
                    "mc_path_cap": MC_PATH_SCHEDULE[-1],
                    "time_metric": "first hit",
                },
                "ground_truths": ground_truth_rows,
                "time_to_accuracy": time_rows,
                "bsde_stability_diagnostic": settling_rows,
                "mc_checkpoints": mc_checkpoint_rows,
            },
            indent=2,
        )
        + "\n"
    )

    summary_lines = [
        "# Phase 3 Block 3 — Formal time-to-accuracy scan",
        "",
        "Primary metric: first-hit wall-clock time. Settling is reported separately.",
        "",
        (
            "Methodology caveat: the MC+CV benchmark measures the same estimator's "
            "internal convergence toward its own high-M reference. It is not an "
            "independent-method validation and raw speed ratios versus Deep BSDE "
            "should not be presented as an unconditional fair-method contest."
        ),
        "",
        "| d | method | target | time | step/paths | actual error | status |",
        "|---:|:---|---:|---:|---:|---:|:---:|",
    ]
    for row in time_rows:
        status = (
            "reached"
            if row["reached_target"]
            else "hit 2e7 cap without reaching"
        )
        summary_lines.append(
            f"| {row['dim']} | {row['method']} | "
            f"<{row['target_relative_error_pct']:.1f}% | "
            f"{float(row['wall_clock_seconds']):.6f} s | "
            f"{row['step_or_paths']} | "
            f"{float(row['actual_relative_error_pct']):.8f}% | {status} |"
        )
    summary_lines.extend(
        [
            "",
            (
                "No MC+CV cell hit the 2e7 path cap."
                if not any(row["hit_path_cap"] for row in time_rows)
                else "Cells marked as path-cap hits are not successful times."
            ),
            "",
            "See `block3_bsde_stability_diagnostic.csv` for settling results; "
            "they are not used as the primary time-to-accuracy metric.",
        ]
    )
    (OUTPUT_DIR / "block3_summary.md").write_text(
        "\n".join(summary_lines) + "\n"
    )
    create_time_plot(
        time_rows,
        OUTPUT_DIR / "time_to_accuracy_vs_dimension_log.png",
    )
    print("Formal Block 3 complete. Stopping before Block 4.", flush=True)


if __name__ == "__main__":
    main()
