"""Run Phase 3 v2 Block 4 Delta (Z0) comparison."""

import csv
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import equation as eqn
from phase3_delta import crn_bump_delta
from run_phase2_error_analysis import (
    BLUE,
    GRID,
    INK,
    MUTED,
    ROSE,
    Config,
    font,
)
from solver import BSDESolver


OUTPUT_DIR = ROOT / "reports" / "phase3" / "block4_delta"
RUN_DIR = OUTPUT_DIR / "bsde_z0_recovery"
BLOCK2_RUN_DIR = (
    ROOT / "reports" / "phase3" / "block2_arithmetic_bsde" / "runs"
)
SEEDS = (5101, 5201, 5202)
SPOT = 100.0
SIGMA = 0.2
DT = 1.0 / 25.0
BUMP_CANDIDATES = (0.01 * SPOT, SIGMA * SPOT * np.sqrt(DT))
D10_BUMP_PATHS = 200_000
MC_REFERENCE_PATHS = 1_000_000
MC_SEED = 6101
MC_ISO_SEED = 6201
MC_ISO_SCHEDULE = (
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


def load_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_history(path: Path, history: np.ndarray) -> None:
    np.savetxt(
        path,
        history,
        fmt=["%d", "%.10e", "%.10e", "%.6f"],
        delimiter=",",
        header="step,loss_function,target_value,elapsed_time",
        comments="",
    )


def recover_float32_z0(seed: int) -> tuple[np.ndarray, float, float]:
    z0_path = RUN_DIR / f"seed_{seed}_z0.npy"
    metadata_path = RUN_DIR / f"seed_{seed}_metadata.json"
    if z0_path.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        return (
            np.load(z0_path),
            float(metadata["elapsed_seconds"]),
            float(metadata["final_y0"]),
        )

    source_config_path = (
        BLOCK2_RUN_DIR
        / f"arithmetic100d_rho_0p0_float32_seed_{seed}_config.json"
    )
    source_history_path = (
        BLOCK2_RUN_DIR
        / f"arithmetic100d_rho_0p0_float32_seed_{seed}_training_history.csv"
    )
    config_dict = load_json(source_config_path)
    if config_dict["net_config"]["dtype"] != "float32":
        raise RuntimeError("Z0 recovery source config is not float32.")

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.set_floatx("float32")
    config = Config(config_dict)
    basket = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    solver = BSDESolver(config, basket)
    logging.info("Recovering float32 Z0 for seed=%d", seed)
    history = solver.train()
    z0 = solver.model.z_init.numpy().astype(np.float64).reshape(-1)
    final_y0 = float(history[-1, 2])
    elapsed = float(history[-1, 3])

    with source_history_path.open(newline="") as source:
        source_rows = list(csv.DictReader(source))
    confirmed_y0 = float(source_rows[-1]["target_value"])
    if abs(final_y0 - confirmed_y0) > 1e-3:
        raise RuntimeError(
            f"Recovered seed {seed} Y0 differs from confirmed run: "
            f"{final_y0} vs {confirmed_y0}."
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.save(z0_path, z0)
    np.savetxt(
        RUN_DIR / f"seed_{seed}_z0.csv",
        z0,
        delimiter=",",
        header="z0",
        comments="",
    )
    save_history(RUN_DIR / f"seed_{seed}_training_history.csv", history)
    metadata_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "dtype": "float32",
                "elapsed_seconds": elapsed,
                "final_y0": final_y0,
                "confirmed_block2_y0": confirmed_y0,
                "source_config": str(source_config_path),
            },
            indent=2,
        )
        + "\n"
    )
    return z0, elapsed, final_y0


def draw_scatter(
    mc_delta: np.ndarray,
    bsde_delta: np.ndarray,
    path: Path,
) -> None:
    width, height = 980, 760
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text(
        (90, 28),
        "Block 4 Component Delta Comparison",
        fill=INK,
        font=font(29, bold=True),
    )
    draw.text(
        (90, 65),
        "100D arithmetic basket; rho=0; each point is one asset",
        fill=MUTED,
        font=font(17),
    )
    left, top, right, bottom = 105, 110, 915, 660
    all_values = np.concatenate([mc_delta, bsde_delta])
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    padding = max((upper - lower) * 0.12, 1e-5)
    lower -= padding
    upper += padding

    def px(value):
        return left + (value - lower) / (upper - lower) * (right - left)

    def py(value):
        return bottom - (value - lower) / (upper - lower) * (bottom - top)

    for index in range(6):
        tick = lower + index * (upper - lower) / 5
        x = px(tick)
        y = py(tick)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{tick:.5f}"
        draw.text((x - 28, bottom + 15), label, fill=MUTED, font=font(13))
        draw.text((20, y - 8), label, fill=MUTED, font=font(13))
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    draw.line((left, top, left, bottom), fill=INK, width=2)
    draw.line(
        (px(lower), py(lower), px(upper), py(upper)),
        fill=ROSE,
        width=3,
    )
    for x_value, y_value in zip(mc_delta, bsde_delta):
        x, y = px(float(x_value)), py(float(y_value))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=BLUE)
    draw.text(
        ((left + right) / 2 - 65, 710),
        "MC+CV Delta",
        fill=INK,
        font=font(18),
    )
    draw.text(
        (105, 88),
        "BSDE Delta",
        fill=INK,
        font=font(16),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "block4_run.log", mode="a"),
        ],
    )

    bump_results = []
    for bump in BUMP_CANDIDATES:
        result = crn_bump_delta(
            dim=10,
            bump=float(bump),
            paths=D10_BUMP_PATHS,
            seed=MC_SEED,
        )
        bump_results.append(result)
        logging.info(
            "d=10 bump=%.6f: Delta mean=%.10f, SD across assets=%.10f, "
            "elapsed=%.4fs",
            bump,
            float(np.mean(result.delta)),
            float(np.std(result.delta, ddof=1)),
            result.elapsed_seconds,
        )
    candidate_rmse = float(
        np.sqrt(
            np.mean(
                (bump_results[0].delta - bump_results[1].delta) ** 2
            )
        )
    )
    mean_ratio = float(
        np.mean(np.abs(bump_results[1].delta))
        / np.mean(np.abs(bump_results[0].delta))
    )
    selected_bump = float(BUMP_CANDIDATES[0])
    if not (0.1 <= mean_ratio <= 10.0):
        raise RuntimeError(
            "Bump candidates differ pathologically; stop for review."
        )
    write_csv(
        OUTPUT_DIR / "bump_selection.csv",
        [
            {
                "bump": result.bump,
                "paths": result.paths,
                "delta_mean": float(np.mean(result.delta)),
                "delta_sd_across_assets": float(
                    np.std(result.delta, ddof=1)
                ),
                "rms_standard_error": float(
                    np.sqrt(np.mean(result.standard_error ** 2))
                ),
                "elapsed_seconds": result.elapsed_seconds,
            }
            for result in bump_results
        ],
        [
            "bump",
            "paths",
            "delta_mean",
            "delta_sd_across_assets",
            "rms_standard_error",
            "elapsed_seconds",
        ],
    )
    logging.info(
        "Bump subtask complete: selected %.6f; candidate RMSE=%.10e, "
        "mean-absolute ratio=%.6f",
        selected_bump,
        candidate_rmse,
        mean_ratio,
    )

    z0_runs = []
    for seed in SEEDS:
        z0, elapsed, final_y0 = recover_float32_z0(seed)
        z0_runs.append(z0)
        logging.info(
            "Z0 subtask seed=%d: mean Z0=%.10f, elapsed=%.1fs, Y0=%.10f",
            seed,
            float(np.mean(z0)),
            elapsed,
            final_y0,
        )
    z0_matrix = np.stack(z0_runs)
    bsde_delta_matrix = z0_matrix / (SIGMA * SPOT)
    bsde_delta_mean = np.mean(bsde_delta_matrix, axis=0)
    bsde_delta_sd = np.std(bsde_delta_matrix, axis=0, ddof=1)
    bsde_delta_sem = bsde_delta_sd / np.sqrt(len(SEEDS))
    target_rms_sem = float(np.sqrt(np.mean(bsde_delta_sem ** 2)))
    bsde_elapsed = sum(
        load_json(RUN_DIR / f"seed_{seed}_metadata.json")[
            "elapsed_seconds"
        ]
        for seed in SEEDS
    )

    iso_rows = []
    iso_result = None
    for paths in MC_ISO_SCHEDULE:
        result = crn_bump_delta(
            dim=100,
            bump=selected_bump,
            paths=paths,
            seed=MC_ISO_SEED,
        )
        rms_se = float(np.sqrt(np.mean(result.standard_error ** 2)))
        iso_rows.append(
            {
                "paths": paths,
                "rms_standard_error": rms_se,
                "target_bsde_rms_sem": target_rms_sem,
                "precision_reached": rms_se <= target_rms_sem,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        logging.info(
            "MC iso-accuracy M=%d: RMS SE=%.10e, target=%.10e, time=%.4fs",
            paths,
            rms_se,
            target_rms_sem,
            result.elapsed_seconds,
        )
        if rms_se <= target_rms_sem:
            iso_result = result
            break
    if iso_result is None:
        raise RuntimeError("MC Delta did not reach the BSDE precision target.")

    mc_reference = crn_bump_delta(
        dim=100,
        bump=selected_bump,
        paths=MC_REFERENCE_PATHS,
        seed=MC_SEED,
    )
    differences = bsde_delta_mean - mc_reference.delta
    rmse = float(np.sqrt(np.mean(differences ** 2)))
    mae = float(np.mean(np.abs(differences)))
    bias = float(np.mean(differences))
    max_abs = float(np.max(np.abs(differences)))
    correlation = float(
        np.corrcoef(mc_reference.delta, bsde_delta_mean)[0, 1]
    )

    component_rows = []
    for index in range(100):
        component_rows.append(
            {
                "asset": index + 1,
                "delta_mc": float(mc_reference.delta[index]),
                "delta_mc_standard_error": float(
                    mc_reference.standard_error[index]
                ),
                "z0_bsde_mean": float(np.mean(z0_matrix[:, index])),
                "delta_bsde_mean": float(bsde_delta_mean[index]),
                "delta_bsde_sd": float(bsde_delta_sd[index]),
                "delta_bsde_sem": float(bsde_delta_sem[index]),
                "difference_bsde_minus_mc": float(differences[index]),
            }
        )
    write_csv(
        OUTPUT_DIR / "delta_components.csv",
        component_rows,
        list(component_rows[0]),
    )
    write_csv(
        OUTPUT_DIR / "iso_accuracy_scan.csv",
        iso_rows,
        list(iso_rows[0]),
    )
    draw_scatter(
        mc_reference.delta,
        bsde_delta_mean,
        OUTPUT_DIR / "delta_mc_vs_bsde_scatter.png",
    )

    timing_rows = [
        {
            "method": "Deep BSDE",
            "precision_metric": "RMS per-asset SEM across 3 seeds",
            "precision_value": target_rms_sem,
            "paths_or_runs": 3,
            "elapsed_seconds": bsde_elapsed,
            "note": "Z0 is a training by-product; elapsed is total 3-seed training time.",
        },
        {
            "method": "MC+CV CRN bump-and-revalue",
            "precision_metric": "RMS per-asset standard error",
            "precision_value": float(
                np.sqrt(np.mean(iso_result.standard_error ** 2))
            ),
            "paths_or_runs": iso_result.paths,
            "elapsed_seconds": iso_result.elapsed_seconds,
            "note": "One 100D CRN run computes all 2x100 bumped prices.",
        },
        {
            "method": "MC+CV CRN high-precision reference",
            "precision_metric": "RMS per-asset standard error",
            "precision_value": float(
                np.sqrt(np.mean(mc_reference.standard_error ** 2))
            ),
            "paths_or_runs": mc_reference.paths,
            "elapsed_seconds": mc_reference.elapsed_seconds,
            "note": "Diagnostic 1e6-path reference; baseline plus 2x100 bumped prices.",
        },
    ]
    write_csv(
        OUTPUT_DIR / "iso_accuracy_timing.csv",
        timing_rows,
        list(timing_rows[0]),
    )

    results = {
        "selected_bump": selected_bump,
        "bump_candidates": {
            "values": [float(value) for value in BUMP_CANDIDATES],
            "candidate_delta_rmse": candidate_rmse,
            "mean_absolute_delta_ratio_bump4_over_bump1": mean_ratio,
            "reason": (
                "No pathological difference; selected default 0.01*S0."
            ),
        },
        "delta_definition": {
            "raw_network_output": "Z0",
            "conversion": "Delta_i = Z0_i / (sigma*S0_i)",
            "sigma_times_spot": SIGMA * SPOT,
        },
        "comparison": {
            "rmse": rmse,
            "mae": mae,
            "bias_bsde_minus_mc": bias,
            "max_absolute_difference": max_abs,
            "correlation": correlation,
        },
        "precision_alignment": {
            "bsde_rms_sem": target_rms_sem,
            "mc_rms_standard_error": timing_rows[1]["precision_value"],
            "bsde_total_seconds": bsde_elapsed,
            "mc_seconds": iso_result.elapsed_seconds,
            "mc_paths": iso_result.paths,
            "mc_reference_seconds": mc_reference.elapsed_seconds,
            "mc_reference_paths": mc_reference.paths,
        },
        "mc_reference": mc_reference.metadata(),
    }
    (OUTPUT_DIR / "block4_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    summary_lines = [
        "# Phase 3 Block 4 — Delta (Z0) comparison",
        "",
        f"Selected bump: {selected_bump:.6f}.",
        (
            f"d=10 candidate RMSE: {candidate_rmse:.10e}; mean-absolute "
            f"Delta ratio: {mean_ratio:.6f}."
        ),
        "",
        (
            "The network output is the BSDE integrand Z0. For dimensional "
            "comparison with bump Delta, Delta=Z0/(sigma*S0)=Z0/20."
        ),
        "",
        "## Difference distribution",
        "",
        f"- RMSE: {rmse:.10e}",
        f"- MAE: {mae:.10e}",
        f"- Bias (BSDE - MC): {bias:.10e}",
        f"- Maximum absolute difference: {max_abs:.10e}",
        f"- Component correlation: {correlation:.8f}",
        "",
        "## Iso-accuracy timing",
        "",
        "| method | precision | work | time |",
        "|:---|---:|---:|---:|",
        (
            f"| Deep BSDE | RMS SEM={target_rms_sem:.10e} | "
            f"3 seeds | {bsde_elapsed:.3f} s |"
        ),
        (
            f"| MC+CV CRN bump | RMS SE="
            f"{timing_rows[1]['precision_value']:.10e} | "
            f"M={iso_result.paths} | {iso_result.elapsed_seconds:.6f} s |"
        ),
        (
            f"| MC+CV CRN high-precision reference | diagnostic | "
            f"M={mc_reference.paths} | {mc_reference.elapsed_seconds:.6f} s |"
        ),
        "",
        (
            "Each MC run evaluates the baseline plus 2x100 bumped prices "
            "(201 valuations), vectorized over one shared CRN path set. "
            "CRN is used for every plus/minus pair. Pathwise differentiation "
            "is not used because the payoff is non-differentiable at the kink "
            "avg(S)=K, where its variance can become unstable."
        ),
    ]
    (OUTPUT_DIR / "block4_summary.md").write_text(
        "\n".join(summary_lines) + "\n"
    )
    logging.info(
        "Block 4 complete: bump=%.6f, RMSE=%.10e, MC iso M=%d. Stopping.",
        selected_bump,
        rmse,
        iso_result.paths,
    )


if __name__ == "__main__":
    main()
