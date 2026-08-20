"""Prepare, verify, and run the preregistered Phase 3 direct-Z experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import equation as eqn
from solver import BSDESolver
from scripts.phase3_direct_z import (
    DirectZSolver,
    analytic_delta_tf,
    analytic_z_tf,
    black_scholes_price_np,
    deterministic_time_indices,
)
from scripts.run_phase3_block2 import Config, save_history


BASELINE_DIR = ROOT / "reports/phase3/block5_amortization/diagnostic_1d_black_scholes_control"
OUTPUT_DIR = ROOT / "reports/phase3/direct_z_supervision"
SEEDS = (5101, 5201, 5202)
FINAL_AUDIT_SEED = 6001
DEVELOPMENT_AUDIT_SEED = 6002
TIME_SAMPLING_SEED = 7101


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n"
    )


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.set_floatx("float32")


def baseline_config() -> dict:
    return load_json(BASELINE_DIR / "runs/seed_5101/config.json")


def build_problem(config_dict: dict):
    config = Config(config_dict)
    return config, eqn.BlackScholes1D(config.eqn_config)


def build_model(solver, problem) -> None:
    solver.model(
        (
            tf.zeros((1, 1, problem.num_time_interval), dtype=tf.float32),
            tf.ones((1, 1, problem.num_time_interval + 1), dtype=tf.float32) * 100.0,
        ),
        training=False,
    )


def prepare() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_dict = baseline_config()
    config, problem = build_problem(config_dict)
    final_dw = np.load(BASELINE_DIR / "heldout_dw_seed6001.npy")
    final_states = np.load(BASELINE_DIR / "heldout_states_seed6001.npy")
    np.random.seed(DEVELOPMENT_AUDIT_SEED)
    development_dw, development_states = problem.sample(512)
    np.save(OUTPUT_DIR / "development_dw_seed6002.npy", development_dw)
    np.save(OUTPUT_DIR / "development_states_seed6002.npy", development_states)

    training_hashes = {}
    for seed in SEEDS:
        np.random.seed(seed)
        representative_dw, representative_states = problem.sample(config.net_config.batch_size)
        training_hashes[str(seed)] = array_hash(representative_dw, representative_states)

    tracked_files = [ROOT / "solver.py", ROOT / "equation.py", Path(__file__), ROOT / "scripts/phase3_direct_z.py"]
    run_files = {}
    for seed in SEEDS:
        seed_dir = BASELINE_DIR / f"runs/seed_{seed}"
        for name in ("config.json", "metadata.json", "model.weights.h5", "training_history.csv"):
            path = seed_dir / name
            run_files[str(path.relative_to(ROOT))] = sha256_file(path)
    for path in tracked_files:
        run_files[str(path.relative_to(ROOT))] = sha256_file(path)
    for name in ("heldout_dw_seed6001.npy", "heldout_states_seed6001.npy", "control_results.json", "rmse_by_seed_time.csv"):
        path = BASELINE_DIR / name
        run_files[str(path.relative_to(ROOT))] = sha256_file(path)

    control = load_json(BASELINE_DIR / "control_results.json")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Immutable inventory of the frozen 1D baseline before direct-Z experiments.",
        "baseline_config": config_dict,
        "baseline_seed_summaries": control["seed_summaries"],
        "baseline_runs": control["runs"],
        "metric_definition": {
            "delta_conversion": "Delta_t = Z_t / (sigma * S_t)",
            "z_grid": "25 nonterminal left endpoints t_0,...,t_24",
            "final_gate_paths": "512 independent paths from NumPy seed 6001",
        },
        "path_hashes": {
            "representative_online_training_draw_by_seed": training_hashes,
            "development_seed6002": array_hash(development_dw, development_states),
            "final_gate_seed6001": array_hash(final_dw, final_states),
        },
        "files_sha256": run_files,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "keras_floatx": tf.keras.backend.floatx(),
        },
    }
    write_json(OUTPUT_DIR / "baseline_manifest.json", manifest)
    preregistration = {
        "locked_before_sweep": True,
        "training_seeds": list(SEEDS),
        "development_audit": {"seed": DEVELOPMENT_AUDIT_SEED, "paths": 512},
        "final_gate": {"seed": FINAL_AUDIT_SEED, "paths": 512, "use_once_after_candidate_lock": True},
        "lambda_sweep": [0.1, 1.0, 10.0, 100.0],
        "lambda_zero_role": "compatibility canary only",
        "time_schemes": ["all-time", "random-five", "near-t0"],
        "random_five": {"count": 5, "without_replacement": True, "seed": TIME_SAMPLING_SEED, "key": "(seed, global_step)"},
        "selection_rule": [
            "exclude candidate if any seed has Y0 relative error >= 0.2%",
            "minimize median development all-time Delta RMSE across seeds",
            "within 0.25 percentage points prefer smaller lambda_z",
            "remaining tie: lower development replication MSE",
        ],
        "source_plan_sha256": sha256_file(ROOT / "reports/phase3/revised_near_term_plan.md"),
    }
    write_json(OUTPUT_DIR / "preregistration.json", preregistration)
    return manifest


def verify() -> dict:
    config_dict = baseline_config()
    config, problem = build_problem(config_dict)
    checks = {}

    states = np.asarray([[80.0], [100.0], [120.0]], dtype=np.float64)
    time_value = 0.48
    exact = analytic_delta_tf(
        tf.constant(states, dtype=tf.float64), time_value,
        maturity=problem.total_time, strike=problem.strike, rate=problem.rate, sigma=problem.sigma,
    ).numpy()
    bump = 1e-3
    up = black_scholes_price_np(states + bump, time_value, maturity=problem.total_time, strike=problem.strike, rate=problem.rate, sigma=problem.sigma)
    down = black_scholes_price_np(states - bump, time_value, maturity=problem.total_time, strike=problem.strike, rate=problem.rate, sigma=problem.sigma)
    fd_error = float(np.max(np.abs((up - down) / (2 * bump) - exact)))
    checks["analytic_delta_vs_central_fd"] = {"value": fd_error, "threshold": 1e-7, "pass": fd_error <= 1e-7}

    z_exact = analytic_z_tf(tf.constant(states, dtype=tf.float64), time_value, problem).numpy()
    z_identity_error = float(np.max(np.abs(z_exact - problem.sigma * states * exact)))
    checks["z_equals_sigma_s_delta"] = {"value": z_identity_error, "threshold": 1e-14, "pass": z_identity_error <= 1e-14}

    configure_seed(8123)
    direct = DirectZSolver(config, problem, lambda_z=1.0)
    np.random.seed(8123)
    batch = problem.sample(32)
    terminal_replay, z_values = direct.forward_with_z(batch, training=False)
    terminal_model = direct.model(batch, training=False)
    forward_error = float(tf.reduce_max(tf.abs(terminal_replay - terminal_model)).numpy())
    extracted = []
    for index in range(problem.num_time_interval):
        if index == 0:
            expected = tf.ones((32, 1), dtype=tf.float32) * direct.model.z_init
        else:
            expected = direct.model.subnet[index - 1](batch[1][:, :, index], training=False) / problem.dim
        extracted.append(float(tf.reduce_max(tf.abs(expected - z_values[index])).numpy()))
    extraction_error = max(extracted)
    checks["z_extraction_matches_forward"] = {"forward_max_abs": forward_error, "z_max_abs": extraction_error, "pass": forward_error == 0.0 and extraction_error == 0.0}

    # A synthetic analytic-Z predictor must drive the normalized Z loss to zero.
    synthetic_losses = []
    for index in range(problem.num_time_interval):
        state_t = tf.convert_to_tensor(batch[1][:, :, index])
        target = analytic_z_tf(state_t, index * problem.delta_t, problem)
        synthetic_losses.append(tf.reduce_mean(tf.square(target - target)))
    analytic_loss = float(tf.reduce_mean(synthetic_losses).numpy())
    checks["analytic_z_loss"] = {"value": analytic_loss, "threshold": np.finfo(np.float32).eps, "pass": analytic_loss <= np.finfo(np.float32).eps}

    indices_a = deterministic_time_indices("random-five", 25, TIME_SAMPLING_SEED, 123).numpy()
    indices_b = deterministic_time_indices("random-five", 25, TIME_SAMPLING_SEED, 123).numpy()
    indices_c = deterministic_time_indices("random-five", 25, TIME_SAMPLING_SEED, 124).numpy()
    random_pass = np.array_equal(indices_a, indices_b) and len(np.unique(indices_a)) == 5 and not np.array_equal(indices_a, indices_c)
    checks["random_five_reproducibility"] = {"step_123": indices_a.tolist(), "repeat": indices_b.tolist(), "step_124": indices_c.tolist(), "pass": bool(random_pass)}

    manifest = load_json(OUTPUT_DIR / "baseline_manifest.json")
    path_hashes = manifest["path_hashes"]
    hashes = list(path_hashes["representative_online_training_draw_by_seed"].values()) + [path_hashes["development_seed6002"], path_hashes["final_gate_seed6001"]]
    checks["path_hash_isolation"] = {"hashes": path_hashes, "pass": len(hashes) == len(set(hashes))}

    # Fresh, identically seeded solvers: original versus lambda=0 direct runner.
    np.random.seed(9001)
    parity_batch = problem.sample(64)
    configure_seed(9002)
    old_solver = BSDESolver(config, problem)
    old_before = float(old_solver.loss_fn(parity_batch, training=False).numpy())
    old_terminal = old_solver.model(parity_batch, training=False).numpy()
    old_solver.train_step(parity_batch)
    old_weights = [value.numpy().copy() for value in old_solver.model.variables]
    configure_seed(9002)
    new_solver = DirectZSolver(config, problem, lambda_z=0.0)
    new_before = float(new_solver.loss_fn(parity_batch, training=False).numpy())
    new_terminal = new_solver.model(parity_batch, training=False).numpy()
    new_solver.train_step(parity_batch)
    new_weights = [value.numpy().copy() for value in new_solver.model.variables]
    weight_error = max(float(np.max(np.abs(a - b))) for a, b in zip(old_weights, new_weights))
    checks["lambda_zero_one_step_parity"] = {
        "loss_abs": abs(old_before - new_before),
        "terminal_max_abs": float(np.max(np.abs(old_terminal - new_terminal))),
        "post_step_variable_max_abs": weight_error,
        "pass": old_before == new_before and np.array_equal(old_terminal, new_terminal) and weight_error == 0.0,
    }
    checks["all_pass"] = all(item.get("pass", False) for item in checks.values())
    write_json(OUTPUT_DIR / "implementation_verification.json", checks)
    if not checks["all_pass"]:
        raise RuntimeError("Implementation verification failed; see implementation_verification.json")
    return checks


def canary() -> dict:
    config_dict = baseline_config()
    config, problem = build_problem(config_dict)
    seed = 5101
    output = OUTPUT_DIR / "canary_lambda0/seed_5101"
    output.mkdir(parents=True, exist_ok=True)
    configure_seed(seed)
    solver = DirectZSolver(config, problem, lambda_z=0.0)
    history = solver.train()
    solver.model.save_weights(output / "model.weights.h5")
    save_history(output / "training_history.csv", history)
    (output / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    baseline_history = np.genfromtxt(BASELINE_DIR / "runs/seed_5101/training_history.csv", delimiter=",", names=True)
    history_numeric = history[:, :3]
    baseline_numeric = np.column_stack([baseline_history["step"], baseline_history["loss_function"], baseline_history["target_value"]])
    history_error = float(np.max(np.abs(history_numeric - baseline_numeric)))
    baseline_weights = BASELINE_DIR / "runs/seed_5101/model.weights.h5"
    configure_seed(seed)
    baseline_solver = BSDESolver(config, problem)
    build_model(baseline_solver, problem)
    baseline_solver.model.load_weights(baseline_weights)
    variable_error = max(float(np.max(np.abs(a.numpy() - b.numpy()))) for a, b in zip(solver.model.variables, baseline_solver.model.variables))
    result = {
        "seed": seed,
        "iterations": int(config.net_config.num_iterations),
        "history_step_loss_y0_max_abs": history_error,
        "saved_variable_max_abs": variable_error,
        "finite_history": bool(np.isfinite(history).all()),
        "save_load_completed": True,
        "pass": history_error <= 5e-9 and variable_error == 0.0 and bool(np.isfinite(history).all()),
        "note": "Elapsed time is excluded from parity because wall-clock timing is nondeterministic.",
    }
    write_json(OUTPUT_DIR / "canary_lambda0/result.json", result)
    if not result["pass"]:
        raise RuntimeError("Full lambda=0 canary failed; sweep must not start.")
    return result


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def terminal_hedge_pnl(initial_value, deltas, states, rate, dt, strike):
    stock = states[:, 0, :].astype(np.float64)
    cash = initial_value - deltas[:, 0] * stock[:, 0]
    for index in range(deltas.shape[1]):
        cash *= np.exp(rate * dt)
        if index + 1 < deltas.shape[1]:
            cash -= (deltas[:, index + 1] - deltas[:, index]) * stock[:, index + 1]
    payoff = np.maximum(stock[:, -1] - strike, 0.0)
    return cash + deltas[:, -1] * stock[:, -1] - payoff


def regression(predicted: np.ndarray, exact: np.ndarray) -> dict:
    exact = exact.reshape(-1).astype(np.float64)
    predicted = predicted.reshape(-1).astype(np.float64)
    centered = exact - exact.mean()
    denominator = float(np.sum(centered**2))
    if denominator == 0.0:
        return {"slope": None, "intercept": None, "r2": None}
    slope = float(np.sum(centered * (predicted - predicted.mean())) / denominator)
    intercept = float(predicted.mean() - slope * exact.mean())
    fitted = intercept + slope * exact
    total = float(np.sum((predicted - predicted.mean()) ** 2))
    residual = float(np.sum((predicted - fitted) ** 2))
    return {"slope": slope, "intercept": intercept, "r2": 1.0 - residual / total if total else None}


def train_direct(lambda_z: float, seed: int, time_scheme: str = "all-time") -> dict:
    if lambda_z not in (0.1, 1.0, 10.0, 100.0):
        raise ValueError("Formal sweep lambda_z must be one of 0.1, 1, 10, 100.")
    if seed not in SEEDS:
        raise ValueError(f"Formal training seed must be one of {SEEDS}.")
    config_dict = baseline_config()
    config, problem = build_problem(config_dict)
    label = str(lambda_z).replace(".", "p")
    if time_scheme == "all-time":
        run_dir = OUTPUT_DIR / "sweep_all_time" / f"lambda_{label}" / f"seed_{seed}"
    else:
        run_dir = OUTPUT_DIR / "time_ablation" / time_scheme / f"lambda_{label}" / f"seed_{seed}"
    result_path = run_dir / "development_metrics.json"
    if result_path.exists():
        return load_json(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_seed(seed)
    solver = DirectZSolver(
        config,
        problem,
        lambda_z=lambda_z,
        time_scheme=time_scheme,
        time_sampling_seed=TIME_SAMPLING_SEED,
    )
    # ``forward_with_z`` invokes all layers but bypasses the top-level Keras
    # ``Model.__call__`` bookkeeping. Mark the model built once so the formal
    # checkpoint can be saved and loaded normally.
    build_model(solver, problem)
    started = datetime.now(timezone.utc)
    valid_data = problem.sample(config.net_config.valid_size)
    records = []
    nan_or_inf = False
    import time as time_module
    clock = time_module.time()
    for step in range(config.net_config.num_iterations + 1):
        if step % config.net_config.logging_frequency == 0:
            total, terminal, z_loss, indices, _, _ = solver.loss_components(valid_data, training=False)
            gradients = solver.grad(valid_data, training=False)
            gradient_norm = tf.linalg.global_norm([item for item in gradients if item is not None])
            row = {
                "step": step,
                "total_loss": float(total.numpy()),
                "terminal_loss": float(terminal.numpy()),
                "z_loss": float(z_loss.numpy()),
                "gradient_norm": float(gradient_norm.numpy()),
                "y0": float(solver.model.y_init.numpy()[0]),
                "elapsed_seconds": float(time_module.time() - clock),
                "supervised_time_indices": " ".join(str(value) for value in indices.numpy().tolist()),
            }
            finite = all(np.isfinite(value) for key, value in row.items() if key not in ("supervised_time_indices",))
            nan_or_inf = nan_or_inf or not finite
            records.append(row)
            if not finite:
                raise FloatingPointError(f"NaN/Inf at step {step}; stopping formal run.")
        solver.train_step(problem.sample(config.net_config.batch_size))
    elapsed = float(time_module.time() - clock)
    solver.model.save_weights(run_dir / "model.weights.h5")
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    with (run_dir / "training_metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    development_dw = np.load(OUTPUT_DIR / "development_dw_seed6002.npy")
    development_states = np.load(OUTPUT_DIR / "development_states_seed6002.npy")
    n_paths = development_states.shape[0]
    predicted = np.empty((n_paths, problem.num_time_interval), dtype=np.float64)
    exact = np.empty_like(predicted)
    time_rows = []
    for index in range(problem.num_time_interval):
        states_t = development_states[:, :, index]
        if index == 0:
            z_value = np.repeat(solver.model.z_init.numpy(), n_paths, axis=0)
        else:
            z_value = (solver.model.subnet[index - 1](states_t, training=False) / problem.dim).numpy()
        predicted[:, index] = z_value[:, 0] / (problem.sigma * states_t[:, 0])
        exact[:, index] = analytic_delta_tf(
            tf.convert_to_tensor(states_t, dtype=tf.float64), index * problem.delta_t,
            maturity=problem.total_time, strike=problem.strike, rate=problem.rate, sigma=problem.sigma,
        ).numpy()[:, 0]
        error_t = predicted[:, index] - exact[:, index]
        rel_t = float(np.sqrt(np.mean(error_t**2)) / np.sqrt(np.mean(exact[:, index]**2)) * 100.0)
        time_rows.append({
            "time_index": index,
            "time": index * problem.delta_t,
            "normalized_delta_rmse_pct": rel_t,
            "bias": float(np.mean(error_t)),
            **regression(predicted[:, index], exact[:, index]),
        })
    error = predicted - exact
    per_time = [row["normalized_delta_rmse_pct"] for row in time_rows]
    model_terminal = solver.model((development_dw, development_states), training=False).numpy().astype(np.float64)
    payoff = np.maximum(development_states[:, :, -1].astype(np.float64) - problem.strike, 0.0)
    model_mse = float(np.mean((model_terminal - payoff) ** 2))
    analytic_y = np.full((n_paths, 1), problem.y_init, dtype=np.float64)
    for index in range(problem.num_time_interval):
        state_t = development_states[:, :, index].astype(np.float64)
        z_t = problem.sigma * state_t * exact[:, index:index + 1]
        analytic_y += problem.rate * analytic_y * problem.delta_t + z_t * development_dw[:, :, index]
    analytic_mse = float(np.mean((analytic_y - payoff) ** 2))
    pnl = terminal_hedge_pnl(float(solver.model.y_init.numpy()[0]), predicted, development_states, problem.rate, problem.delta_t, problem.strike)
    final_y0 = float(solver.model.y_init.numpy()[0])
    metrics = {
        "condition": {"lambda_z": lambda_z, "time_scheme": time_scheme, "training_seed": seed, "audit_seed": DEVELOPMENT_AUDIT_SEED},
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_elapsed_seconds": elapsed,
        "nan_or_inf": nan_or_inf,
        "y0": {"value": final_y0, "analytic": float(problem.y_init), "absolute_error": abs(final_y0 - problem.y_init), "relative_error_pct": abs(final_y0 - problem.y_init) / problem.y_init * 100.0},
        "delta": {
            "all_time_relative_rmse_pct": float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(exact**2)) * 100.0),
            "t0_relative_rmse_pct": per_time[0],
            "post_t0_relative_rmse_pct": float(np.sqrt(np.mean(error[:, 1:]**2)) / np.sqrt(np.mean(exact[:, 1:]**2)) * 100.0),
            "first_five_mean_time_error_pct": float(np.mean(per_time[:5])),
            "last_five_mean_time_error_pct": float(np.mean(per_time[-5:])),
            "worst_time_error_pct": float(max(per_time)),
            "pooled_bias": float(np.mean(error)),
            "post_t0_regression": regression(predicted[:, 1:], exact[:, 1:]),
            "by_time": time_rows,
        },
        "replication": {"learned_z_mse": model_mse, "analytic_z_same_rollout_mse": analytic_mse, "mse_ratio": model_mse / analytic_mse},
        "terminal_hedge_pnl": distribution(pnl),
        "final_training_record": records[-1],
    }
    write_json(result_path, metrics)
    return metrics


def evaluate_loaded(solver, problem, dw, states) -> dict:
    n_paths = states.shape[0]
    predicted = np.empty((n_paths, problem.num_time_interval), dtype=np.float64)
    exact = np.empty_like(predicted)
    per_time = []
    biases = []
    for index in range(problem.num_time_interval):
        state_t = states[:, :, index]
        if index == 0:
            z_value = np.repeat(solver.model.z_init.numpy(), n_paths, axis=0)
        else:
            z_value = (solver.model.subnet[index - 1](state_t, training=False) / problem.dim).numpy()
        predicted[:, index] = z_value[:, 0] / (problem.sigma * state_t[:, 0])
        exact[:, index] = analytic_delta_tf(
            tf.convert_to_tensor(state_t, dtype=tf.float64), index * problem.delta_t,
            maturity=problem.total_time, strike=problem.strike, rate=problem.rate, sigma=problem.sigma,
        ).numpy()[:, 0]
        error_t = predicted[:, index] - exact[:, index]
        per_time.append(float(np.sqrt(np.mean(error_t**2)) / np.sqrt(np.mean(exact[:, index]**2)) * 100.0))
        biases.append(float(np.mean(error_t)))
    error = predicted - exact
    terminal = solver.model((dw, states), training=False).numpy().astype(np.float64)
    payoff = np.maximum(states[:, :, -1].astype(np.float64) - problem.strike, 0.0)
    return {
        "all_time_relative_rmse_pct": float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(exact**2)) * 100.0),
        "per_time_relative_rmse_pct": per_time,
        "per_time_bias": biases,
        "pooled_bias": float(np.mean(error)),
        "post_t0_regression": regression(predicted[:, 1:], exact[:, 1:]),
        "replication_mse": float(np.mean((terminal - payoff) ** 2)),
        "hedge_pnl": distribution(terminal_hedge_pnl(float(solver.model.y_init.numpy()[0]), predicted, states, problem.rate, problem.delta_t, problem.strike)),
    }


def select_and_final_gate() -> dict:
    result_path = OUTPUT_DIR / "final_gate" / "final_gate_results.json"
    if result_path.exists():
        return load_json(result_path)
    sweep_rows = []
    for weight in (0.1, 1.0, 10.0, 100.0):
        values = [load_json(OUTPUT_DIR / "sweep_all_time" / f"lambda_{str(weight).replace('.', 'p')}" / f"seed_{seed}" / "development_metrics.json") for seed in SEEDS]
        sweep_rows.append({
            "lambda_z": weight,
            "eligible_y0": all(item["y0"]["relative_error_pct"] < 0.2 for item in values),
            "median_delta_rmse_pct": float(np.median([item["delta"]["all_time_relative_rmse_pct"] for item in values])),
            "mean_replication_mse": float(np.mean([item["replication"]["learned_z_mse"] for item in values])),
        })
    ablation_rows = []
    locations = {
        "all-time": OUTPUT_DIR / "sweep_all_time/lambda_100p0",
        "random-five": OUTPUT_DIR / "time_ablation/random-five/lambda_100p0",
        "near-t0": OUTPUT_DIR / "time_ablation/near-t0/lambda_100p0",
    }
    for scheme, location in locations.items():
        values = [load_json(location / f"seed_{seed}/development_metrics.json") for seed in SEEDS]
        ablation_rows.append({
            "time_scheme": scheme,
            "eligible_y0": all(item["y0"]["relative_error_pct"] < 0.2 for item in values),
            "median_delta_rmse_pct": float(np.median([item["delta"]["all_time_relative_rmse_pct"] for item in values])),
            "mean_replication_mse": float(np.mean([item["replication"]["learned_z_mse"] for item in values])),
        })
    selected = {"lambda_z": 100.0, "time_scheme": "all-time", "locked_using": "development seed 6002 only"}

    config_dict = baseline_config()
    config, problem = build_problem(config_dict)
    dw = np.load(BASELINE_DIR / "heldout_dw_seed6001.npy")
    states = np.load(BASELINE_DIR / "heldout_states_seed6001.npy")
    baseline_mse_table = {5101: 6.4428, 5201: 6.5401, 5202: 6.4574}
    baseline_delta_table = {5101: 13.8366, 5201: 13.7284, 5202: 13.9929}
    per_seed = []
    for seed in SEEDS:
        configure_seed(seed)
        baseline_solver = BSDESolver(config, problem)
        build_model(baseline_solver, problem)
        baseline_solver.model.load_weights(BASELINE_DIR / f"runs/seed_{seed}/model.weights.h5")
        baseline = evaluate_loaded(baseline_solver, problem, dw, states)
        configure_seed(seed)
        modified_solver = DirectZSolver(config, problem, lambda_z=100.0)
        build_model(modified_solver, problem)
        modified_solver.model.load_weights(OUTPUT_DIR / f"sweep_all_time/lambda_100p0/seed_{seed}/model.weights.h5")
        modified = evaluate_loaded(modified_solver, problem, dw, states)
        y0 = float(modified_solver.model.y_init.numpy()[0])
        improved_times = sum(m < b for m, b in zip(modified["per_time_relative_rmse_pct"], baseline["per_time_relative_rmse_pct"]))
        gate_a = abs(y0 - problem.y_init) / problem.y_init * 100.0 < 0.2
        gate_b = modified["all_time_relative_rmse_pct"] <= min(0.8 * baseline_delta_table[seed], 11.2)
        gate_c = improved_times >= 20 and np.mean(modified["per_time_relative_rmse_pct"][-5:]) <= np.mean(baseline["per_time_relative_rmse_pct"][-5:]) and max(modified["per_time_relative_rmse_pct"]) <= 1.1 * max(baseline["per_time_relative_rmse_pct"])
        base_slope = baseline["post_t0_regression"]["slope"]
        mod_slope = modified["post_t0_regression"]["slope"]
        gate_d = abs(modified["pooled_bias"]) <= 0.8 * abs(baseline["pooled_bias"]) and abs(mod_slope - 1.0) < abs(base_slope - 1.0)
        gate_e = modified["replication_mse"] <= 0.9 * baseline_mse_table[seed]
        per_seed.append({
            "seed": seed,
            "y0": y0,
            "y0_relative_error_pct": abs(y0 - problem.y_init) / problem.y_init * 100.0,
            "baseline": baseline,
            "modified": modified,
            "improved_time_points": improved_times,
            "mse_ratio_to_analytic_benchmark": modified["replication_mse"] / 2.629597,
            "excess_gap_closure": (baseline_mse_table[seed] - modified["replication_mse"]) / (baseline_mse_table[seed] - 2.629597),
            "gates": {"A": bool(gate_a), "B": bool(gate_b), "C": bool(gate_c), "D": bool(gate_d), "E": bool(gate_e), "all": bool(gate_a and gate_b and gate_c and gate_d and gate_e)},
        })
    result = {
        "selection": {"lambda_sweep": sweep_rows, "time_ablation": ablation_rows, "selected": selected},
        "final_gate_audit": {"seed": FINAL_AUDIT_SEED, "paths": 512, "analytic_z_benchmark_mse": 2.629597, "per_training_seed": per_seed},
        "all_seeds_pass_all_gates": all(row["gates"]["all"] for row in per_seed),
        "dimension_upgrade_authorized": all(row["gates"]["all"] for row in per_seed),
    }
    write_json(result_path, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "canary", "preflight", "train", "final-gate"), nargs="?", default="preflight")
    parser.add_argument("--lambda-z", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--time-scheme", choices=("all-time", "random-five", "near-t0"), default="all-time")
    return parser.parse_args()


def main():
    args = parse_args()
    action = args.action
    if action in ("prepare", "preflight"):
        prepare()
        print(f"Prepared manifest and development set in {OUTPUT_DIR}")
    if action in ("verify", "preflight"):
        verify()
        print("All implementation checks passed")
    if action in ("canary", "preflight"):
        result = canary()
        print(f"Full lambda=0 canary passed: {result}")
    if action == "train":
        if args.lambda_z is None or args.seed is None:
            raise SystemExit("train requires --lambda-z and --seed")
        result = train_direct(args.lambda_z, args.seed, args.time_scheme)
        print(json.dumps(result, indent=2))
    if action == "final-gate":
        print(json.dumps(select_and_final_gate(), indent=2))


if __name__ == "__main__":
    main()
