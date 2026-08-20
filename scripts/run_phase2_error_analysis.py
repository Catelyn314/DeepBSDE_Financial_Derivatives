"""Run Phase 2 error analysis experiments and generate report assets."""

import argparse
import copy
import csv
import json
import logging
import random
import sys
import time
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports" / "phase2_error_analysis"
BASE_BASKET_CONFIG = ROOT / "configs" / "geometric_basket_100d.json"
BASE_BS_CONFIG = ROOT / "configs" / "bs_1d.json"
FEEDBACK_DIR = OUT_DIR / "prof_feedback_2026_07_17"

WHITE = (255, 255, 255)
INK = (31, 41, 55)
MUTED = (107, 114, 128)
GRID = (229, 231, 235)
BLUE = (37, 99, 235)
ROSE = (225, 29, 72)
GREEN = (5, 150, 105)
AMBER = (217, 119, 6)
PURPLE = (124, 58, 237)
CYAN = (8, 145, 178)
COLORS = [BLUE, GREEN, AMBER, PURPLE, CYAN, (220, 38, 38), (79, 70, 229)]


class DictToObject:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def to_dict(self):
        return self._dict


class Config:
    def __init__(self, config_dict):
        self.eqn_config = DictToObject(config_dict["eqn_config"])
        self.net_config = DictToObject(config_dict["net_config"])
        self._original_dict = config_dict

    def to_dict(self):
        return self._original_dict


def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call_price(spot, strike, rate, sigma, maturity):
    d1 = (log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return spot * normal_cdf(d1) - strike * exp(-rate * maturity) * normal_cdf(d2)


def geometric_basket_call_price(spot, strike, rate, sigma, maturity, dim, rho=0.0):
    variance = sigma ** 2 * maturity * (1.0 + (dim - 1) * rho) / dim
    std = sqrt(variance)
    log_mean = log(spot) + (rate - 0.5 * sigma ** 2) * maturity
    d2 = (log_mean - log(strike)) / std
    d1 = d2 + std
    discounted_forward_moment = exp(-rate * maturity + log_mean + 0.5 * variance)
    discounted_strike = strike * exp(-rate * maturity)
    return discounted_forward_moment * normal_cdf(d1) - discounted_strike * normal_cdf(d2)


def load_json(path):
    with path.open() as config_file:
        return json.load(config_file)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_history(path):
    with path.open(newline="") as history_file:
        return [
            {
                "step": int(row["step"]),
                "loss": float(row["loss_function"]),
                "y0": float(row["target_value"]),
                "elapsed": float(row["elapsed_time"]),
            }
            for row in csv.DictReader(history_file)
        ]


def save_history(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        history,
        fmt=["%d", "%.5e", "%.5e", "%d"],
        delimiter=",",
        header="step,loss_function,target_value,elapsed_time",
        comments="",
    )


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def nice_ticks(vmin, vmax, count=5):
    if vmin == vmax:
        return [vmin]
    return [vmin + i * (vmax - vmin) / (count - 1) for i in range(count)]


def chart_canvas(title, subtitle):
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((105, 28), title, fill=INK, font=font(30, bold=True))
    draw.text((105, 64), subtitle, fill=MUTED, font=font(18))
    return image, draw, (105, 95, width - 55, height - 95)


def project_point(x, y, x_min, x_max, y_min, y_max, bounds):
    left, top, right, bottom = bounds
    px = left + (x - x_min) / (x_max - x_min) * (right - left)
    py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
    return px, py


def draw_axes(draw, bounds, x_min, x_max, y_min, y_max, x_label):
    left, top, right, bottom = bounds
    tick_font = font(15)
    label_font = font(18)
    for tick in nice_ticks(x_min, x_max):
        x = left + (tick - x_min) / (x_max - x_min) * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = f"{int(tick)}" if abs(tick) >= 1 else f"{tick:.2g}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 18), label, fill=MUTED, font=tick_font)
    for tick in nice_ticks(y_min, y_max):
        y = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{tick:.4g}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 16 - (bbox[2] - bbox[0]), y - 9), label, fill=MUTED, font=tick_font)
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    draw.line((left, top, left, bottom), fill=INK, width=2)
    bbox = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((left + right) / 2 - (bbox[2] - bbox[0]) / 2, 714), x_label, fill=INK, font=label_font)


def draw_line_chart(series, title, subtitle, output_path, analytic=None, x_label="Step"):
    all_points = [point for item in series for point in item["points"]]
    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points] + ([] if analytic is None else [analytic])
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
    y_min -= padding
    y_max += padding
    image, draw, bounds = chart_canvas(title, subtitle)
    draw_axes(draw, bounds, x_min, x_max, y_min, y_max, x_label)

    legend_x, legend_y = bounds[2] - 335, 42
    for idx, item in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        points = [project_point(x, y, x_min, x_max, y_min, y_max, bounds) for x, y in item["points"]]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points[::max(1, len(points) // 12)]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        ly = legend_y + idx * 28
        draw.line((legend_x, ly + 11, legend_x + 44, ly + 11), fill=color, width=4)
        draw.text((legend_x + 56, ly), item["label"], fill=INK, font=font(17))

    if analytic is not None:
        y = project_point(x_min, analytic, x_min, x_max, y_min, y_max, bounds)[1]
        draw.line((bounds[0], y, bounds[2], y), fill=ROSE, width=3)
        ly = legend_y + len(series) * 28
        draw.line((legend_x, ly + 11, legend_x + 44, ly + 11), fill=ROSE, width=3)
        draw.text((legend_x + 56, ly), f"Analytic = {analytic:.6f}", fill=INK, font=font(17))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_scatter(rows, title, subtitle, output_path, x_key, y_key, analytic=None, x_label="Seed"):
    x_values = [float(row[x_key]) for row in rows]
    y_values = [float(row[y_key]) for row in rows] + ([] if analytic is None else [analytic])
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_min, y_max = min(y_values), max(y_values)
    padding = (y_max - y_min) * 0.18 if y_max > y_min else 1.0
    y_min -= padding
    y_max += padding
    image, draw, bounds = chart_canvas(title, subtitle)
    draw_axes(draw, bounds, x_min, x_max, y_min, y_max, x_label)
    for row in rows:
        x, y = project_point(float(row[x_key]), float(row[y_key]), x_min, x_max, y_min, y_max, bounds)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BLUE)
    if analytic is not None:
        y = project_point(x_min, analytic, x_min, x_max, y_min, y_max, bounds)[1]
        draw.line((bounds[0], y, bounds[2], y), fill=ROSE, width=3)
        draw.text((bounds[2] - 280, 45), f"Analytic = {analytic:.6f}", fill=INK, font=font(17))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_metric_chart(rows, title, subtitle, output_path, x_key, y_key, x_label, log_x=False, log_y=False):
    points = []
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        points.append((log(x) if log_x else x, log(y) if log_y else y))
    draw_line_chart(
        [{"label": y_key.replace("_", " "), "points": points}],
        title,
        subtitle,
        output_path,
        x_label=x_label,
    )


def transform_axis_value(value, log_scale):
    return log(float(value)) if log_scale else float(value)


def draw_errorbar_chart(rows, title, subtitle, output_path, x_key, mean_key, std_key, x_label, y_label,
                        log_x=False, log_y=False):
    valid_rows = [row for row in rows if row.get(mean_key) not in ("", None)]
    x_values = [transform_axis_value(row[x_key], log_x) for row in valid_rows]
    means = [transform_axis_value(row[mean_key], log_y) for row in valid_rows]
    lower_values = [
        transform_axis_value(max(float(row[mean_key]) - float(row.get(std_key, 0.0)), 1e-12), log_y)
        for row in valid_rows
    ]
    upper_values = [
        transform_axis_value(float(row[mean_key]) + float(row.get(std_key, 0.0)), log_y)
        for row in valid_rows
    ]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_min, y_max = min(lower_values), max(upper_values)
    padding = (y_max - y_min) * 0.18 if y_max > y_min else 1.0
    y_min -= padding
    y_max += padding

    image, draw, bounds = chart_canvas(title, subtitle)
    left, top, right, bottom = bounds
    tick_font = font(15)
    label_font = font(18)

    x_ticks = [float(row[x_key]) for row in valid_rows]
    for tick in x_ticks:
        x = project_point(transform_axis_value(tick, log_x), y_min, x_min, x_max, y_min, y_max, bounds)[0]
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = f"{int(tick)}" if tick >= 1 else f"{tick:.2g}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 18), label, fill=MUTED, font=tick_font)
    for tick in nice_ticks(y_min, y_max):
        y = project_point(x_min, tick, x_min, x_max, y_min, y_max, bounds)[1]
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{exp(tick):.4g}" if log_y else f"{tick:.4g}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 16 - (bbox[2] - bbox[0]), y - 9), label, fill=MUTED, font=tick_font)
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    draw.line((left, top, left, bottom), fill=INK, width=2)
    xlabel = x_label
    bbox = draw.textbbox((0, 0), xlabel, font=label_font)
    draw.text(((left + right) / 2 - (bbox[2] - bbox[0]) / 2, 714), xlabel, fill=INK, font=label_font)

    points = []
    for row, mean, low, high in zip(valid_rows, means, lower_values, upper_values):
        x = transform_axis_value(row[x_key], log_x)
        px, py = project_point(x, mean, x_min, x_max, y_min, y_max, bounds)
        _, p_low = project_point(x, low, x_min, x_max, y_min, y_max, bounds)
        _, p_high = project_point(x, high, x_min, x_max, y_min, y_max, bounds)
        draw.line((px, p_high, px, p_low), fill=BLUE, width=3)
        draw.line((px - 9, p_high, px + 9, p_high), fill=BLUE, width=3)
        draw.line((px - 9, p_low, px + 9, p_low), fill=BLUE, width=3)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=BLUE)
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill=BLUE, width=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_mean_band_chart(
    series,
    title,
    subtitle,
    output_path,
    analytic=None,
    x_label="Step",
    reference_label="Analytic",
):
    all_points = []
    for item in series:
        all_points.extend([(step, mean) for step, mean, _, _ in item["points"]])
        all_points.extend([(step, low) for step, _, low, _ in item["points"]])
        all_points.extend([(step, high) for step, _, _, high in item["points"]])
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points] + ([] if analytic is None else [analytic])
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
    y_min -= padding
    y_max += padding
    image, draw, bounds = chart_canvas(title, subtitle)
    draw_axes(draw, bounds, x_min, x_max, y_min, y_max, x_label)

    legend_x, legend_y = bounds[2] - 340, 42
    for idx, item in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        fill_color = tuple(int(0.84 * 255 + 0.16 * channel) for channel in color)
        upper = [project_point(step, high, x_min, x_max, y_min, y_max, bounds) for step, _, _, high in item["points"]]
        lower = [project_point(step, low, x_min, x_max, y_min, y_max, bounds) for step, _, low, _ in reversed(item["points"])]
        if len(upper) > 1:
            draw.polygon(upper + lower, fill=fill_color)
        mean_points = [project_point(step, mean, x_min, x_max, y_min, y_max, bounds) for step, mean, _, _ in item["points"]]
        draw.line(mean_points, fill=color, width=4, joint="curve")
        ly = legend_y + idx * 28
        draw.line((legend_x, ly + 11, legend_x + 44, ly + 11), fill=color, width=4)
        draw.text((legend_x + 56, ly), item["label"], fill=INK, font=font(17))

    if analytic is not None:
        y = project_point(x_min, analytic, x_min, x_max, y_min, y_max, bounds)[1]
        draw.line((bounds[0], y, bounds[2], y), fill=ROSE, width=3)
        ly = legend_y + len(series) * 28
        draw.line((legend_x, ly + 11, legend_x + 44, ly + 11), fill=ROSE, width=3)
        draw.text(
            (legend_x + 56, ly),
            f"{reference_label} = {analytic:.6f}",
            fill=INK,
            font=font(17),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_config(base_config, eqn_updates=None, net_updates=None):
    config = copy.deepcopy(base_config)
    for key, value in (eqn_updates or {}).items():
        config["eqn_config"][key] = value
    for key, value in (net_updates or {}).items():
        config["net_config"][key] = value
    config["net_config"]["verbose"] = False
    return config


def analytic_price(config_dict):
    eqn_config = config_dict["eqn_config"]
    if eqn_config["eqn_name"] == "BlackScholes1D":
        return black_scholes_call_price(
            eqn_config["x_init"], eqn_config["strike"], eqn_config["rate"],
            eqn_config["sigma"], eqn_config["total_time"])
    rho = eqn_config.get("rho", 0.0)
    return geometric_basket_call_price(
        eqn_config["x_init"], eqn_config["strike"], eqn_config["rate"],
        eqn_config["sigma"], eqn_config["total_time"], eqn_config["dim"], rho)


def run_training(config_dict, run_dir, run_name, seed):
    import tensorflow as tf

    import equation as eqn
    from solver import BSDESolver

    history_path = run_dir / f"{run_name}_training_history.csv"
    config_path = run_dir / f"{run_name}_config.json"
    if history_path.exists():
        history = read_history(history_path)
        return history, analytic_price(config_dict)

    run_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as config_file:
        json.dump(config_dict, config_file, indent=2)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.set_floatx(config_dict["net_config"]["dtype"])

    config = Config(config_dict)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    solver = BSDESolver(config, bsde)
    start = time.time()
    history = solver.train()
    elapsed = time.time() - start
    save_history(history_path, history)
    print(f"finished {run_name}: Y0={history[-1, 2]:.6f}, elapsed={elapsed:.1f}s")
    return read_history(history_path), bsde.y_init


def summarize_final(name, history, analytic, extra=None):
    final = history[-1]
    abs_error = abs(final["y0"] - analytic)
    row = {
        "name": name,
        "analytic_price": analytic,
        "final_y0": final["y0"],
        "final_loss": final["loss"],
        "absolute_error": abs_error,
        "relative_error_pct": abs_error / analytic * 100.0,
        "elapsed_time_s": final["elapsed"],
    }
    if extra:
        row.update(extra)
    return row


def rel_error_pct(y0, analytic):
    return abs(float(y0) - float(analytic)) / float(analytic) * 100.0


def first_step_under(history, analytic, threshold_pct):
    for row in history:
        rel_error = abs(row["y0"] - analytic) / analytic * 100.0
        if rel_error < threshold_pct:
            return row["step"]
    return ""


def settling_step(history, analytic, threshold_pct):
    rel_errors = [rel_error_pct(row["y0"], analytic) for row in history]
    steps = [row["step"] for row in history]
    above_indices = [idx for idx, value in enumerate(rel_errors) if value > threshold_pct]
    if not above_indices:
        return steps[0] if steps else ""
    last_above = above_indices[-1]
    if last_above == len(steps) - 1:
        return ""
    return steps[last_above + 1]


def y0_std_in_window(history, start_step, end_step):
    values = [row["y0"] for row in history if start_step <= row["step"] <= end_step]
    if not values:
        return ""
    return stats(values)[1] if len(values) > 1 else 0.0


def experiment_seed_variance():
    exp_dir = OUT_DIR / "experiment1_seed_variance"
    base_basket = load_json(BASE_BASKET_CONFIG)
    base_bs = load_json(BASE_BS_CONFIG)
    rows = []

    basket_seeds = list(range(1001, 1009))
    for seed in basket_seeds:
        config = make_config(base_basket)
        run_name = f"basket100d_seed_{seed}"
        print(f"running {run_name}")
        history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
        rows.append(summarize_final("100D Geometric Basket", history, analytic, {"seed": seed, "problem": "100D"}))

    bs_rows = []
    for seed in range(2001, 2006):
        config = make_config(base_bs)
        run_name = f"bs1d_seed_{seed}"
        print(f"running {run_name}")
        history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
        bs_rows.append(summarize_final("1D Black-Scholes", history, analytic, {"seed": seed, "problem": "1D"}))

    fieldnames = ["problem", "seed", "name", "analytic_price", "final_y0", "final_loss", "absolute_error", "relative_error_pct", "elapsed_time_s"]
    write_csv(exp_dir / "seed_variance_100d.csv", rows, fieldnames)
    write_csv(exp_dir / "seed_variance_1d.csv", bs_rows, fieldnames)

    analytic_100d = rows[0]["analytic_price"]
    draw_scatter(rows, "Seed Variance: 100D Geometric Basket", "Final Y0 across independent random seeds", exp_dir / "seed_variance_100d_y0.png", "seed", "final_y0", analytic_100d)
    draw_scatter(bs_rows, "Seed Variance: 1D Black-Scholes", "Final Y0 across independent random seeds", exp_dir / "seed_variance_1d_y0.png", "seed", "final_y0", bs_rows[0]["analytic_price"])
    return rows, bs_rows


def experiment_time_discretization():
    exp_dir = OUT_DIR / "experiment2_time_discretization"
    base = load_json(BASE_BASKET_CONFIG)
    rows = []
    seed = 3101
    for num_steps in [25, 50, 100, 200, 400]:
        config = make_config(base, {"num_time_interval": num_steps})
        run_name = f"basket100d_N_{num_steps}"
        print(f"running {run_name}")
        history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
        rows.append(summarize_final("100D Geometric Basket", history, analytic, {"num_time_interval": num_steps, "seed": seed}))
    fieldnames = ["num_time_interval", "seed", "name", "analytic_price", "final_y0", "final_loss", "absolute_error", "relative_error_pct", "elapsed_time_s"]
    write_csv(exp_dir / "time_discretization.csv", rows, fieldnames)
    draw_metric_chart(rows, "Time Discretization: Relative Error vs N", "Log-log view of final pricing error", exp_dir / "relative_error_vs_N.png", "num_time_interval", "relative_error_pct", "log(N)", log_x=True, log_y=True)
    draw_metric_chart(rows, "Time Discretization: Training Time vs N", "Wall-clock cost for one training run", exp_dir / "training_time_vs_N.png", "num_time_interval", "elapsed_time_s", "N")
    return rows


def collect_time_discretization():
    """Summarize completed time-discretization runs without launching missing ones."""
    exp_dir = OUT_DIR / "experiment2_time_discretization"
    base = load_json(BASE_BASKET_CONFIG)
    rows = []
    completed_rows = []
    seed = 3101
    for num_steps in [25, 50, 100, 200, 400]:
        config = make_config(base, {"num_time_interval": num_steps})
        analytic = analytic_price(config)
        history_path = exp_dir / "runs" / f"basket100d_N_{num_steps}_training_history.csv"
        if history_path.exists():
            history = read_history(history_path)
            row = summarize_final(
                "100D Geometric Basket",
                history,
                analytic,
                {"num_time_interval": num_steps, "seed": seed, "status": "completed", "note": ""},
            )
            rows.append(row)
            completed_rows.append(row)
        else:
            rows.append({
                "num_time_interval": num_steps,
                "seed": seed,
                "status": "not_completed",
                "note": "Stopped because nonshared time-layer model made this run too slow in the local session.",
                "name": "100D Geometric Basket",
                "analytic_price": analytic,
                "final_y0": "",
                "final_loss": "",
                "absolute_error": "",
                "relative_error_pct": "",
                "elapsed_time_s": "",
            })
    fieldnames = [
        "num_time_interval", "seed", "status", "note", "name", "analytic_price", "final_y0",
        "final_loss", "absolute_error", "relative_error_pct", "elapsed_time_s",
    ]
    write_csv(exp_dir / "time_discretization.csv", rows, fieldnames)
    if completed_rows:
        draw_metric_chart(
            completed_rows,
            "Time Discretization: Relative Error vs N",
            "Completed runs only; N=200,400 exceeded local runtime budget",
            exp_dir / "relative_error_vs_N.png",
            "num_time_interval",
            "relative_error_pct",
            "log(N)",
            log_x=True,
            log_y=True,
        )
        draw_metric_chart(
            completed_rows,
            "Time Discretization: Training Time vs N",
            "Completed runs only; cost rises quickly for the nonshared solver",
            exp_dir / "training_time_vs_N.png",
            "num_time_interval",
            "elapsed_time_s",
            "N",
        )
    return rows


def experiment_batch_sensitivity():
    exp_dir = OUT_DIR / "experiment3_batch_optimization"
    base = load_json(BASE_BASKET_CONFIG)
    seed = 4101
    rows = []
    convergence_series = []
    for batch_size in [64, 128, 256, 512]:
        config = make_config(base, net_updates={"batch_size": batch_size})
        run_name = f"basket100d_batch_{batch_size}"
        print(f"running {run_name}")
        history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
        row = summarize_final("100D Geometric Basket", history, analytic, {"batch_size": batch_size, "seed": seed, "max_iterations": config["net_config"]["num_iterations"]})
        row["first_step_rel_error_lt_0_5_pct"] = first_step_under(history, analytic, 0.5)
        rows.append(row)
        convergence_series.append({"label": f"batch {batch_size}", "points": [(item["step"], item["y0"]) for item in history]})

    config = make_config(base, net_updates={"batch_size": 64, "num_iterations": 12000, "lr_boundaries": [4000, 8000]})
    run_name = "basket100d_batch_64_long"
    print(f"running {run_name}")
    history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
    row = summarize_final("100D Geometric Basket", history, analytic, {"batch_size": 64, "seed": seed, "max_iterations": 12000})
    row["first_step_rel_error_lt_0_5_pct"] = first_step_under(history, analytic, 0.5)
    rows.append(row)
    convergence_series.append({"label": "batch 64 long", "points": [(item["step"], item["y0"]) for item in history]})

    fieldnames = [
        "batch_size", "seed", "max_iterations", "first_step_rel_error_lt_0_5_pct", "name",
        "analytic_price", "final_y0", "final_loss", "absolute_error", "relative_error_pct", "elapsed_time_s",
    ]
    write_csv(exp_dir / "batch_optimization_sensitivity.csv", rows, fieldnames)
    draw_line_chart(
        convergence_series,
        "Batch Size Sensitivity: Y0 Convergence",
        "Fixed N=20; analytic reference line included",
        exp_dir / "batch_size_y0_convergence.png",
        analytic=analytic,
    )
    return rows


def summarize_group(rows, group_key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    summary = []
    for group, items in grouped.items():
        y0_mean, y0_std = stats([item["final_y0"] for item in items])
        err_mean, err_std = stats([item["relative_error_pct"] for item in items])
        time_mean, time_std = stats([item["elapsed_time_s"] for item in items])
        summary.append({
            group_key: group,
            "num_runs": len(items),
            "analytic_price": items[0]["analytic_price"],
            "mean_y0": y0_mean,
            "std_y0": y0_std,
            "mean_relative_error_pct": err_mean,
            "std_relative_error_pct": err_std,
            "mean_elapsed_time_s": time_mean,
            "std_elapsed_time_s": time_std,
        })
    return sorted(summary, key=lambda row: float(row[group_key]))


def experiment_time_discretization_rerun():
    exp_dir = OUT_DIR / "experiment2_time_discretization" / "rerun"
    base = load_json(BASE_BASKET_CONFIG)
    rows = []
    seed_plan = {
        25: [3101, 3201, 3202],
        50: [3101, 3201, 3202],
        100: [3101, 3201, 3202],
    }

    for num_steps, seeds in seed_plan.items():
        for seed in seeds:
            config = make_config(base, {"num_time_interval": num_steps})
            if seed == 3101:
                source_history = OUT_DIR / "experiment2_time_discretization" / "runs" / f"basket100d_N_{num_steps}_training_history.csv"
                if source_history.exists():
                    history = read_history(source_history)
                    analytic = analytic_price(config)
                else:
                    history, analytic = run_training(config, exp_dir / "runs", f"basket100d_N_{num_steps}_seed_{seed}", seed)
            else:
                run_name = f"basket100d_N_{num_steps}_seed_{seed}"
                print(f"running {run_name}")
                history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
            rows.append(summarize_final(
                "100D Geometric Basket",
                history,
                analytic,
                {"num_time_interval": num_steps, "seed": seed},
            ))

    per_seed_fields = [
        "num_time_interval", "seed", "name", "analytic_price", "final_y0", "final_loss",
        "absolute_error", "relative_error_pct", "elapsed_time_s",
    ]
    write_csv(exp_dir / "time_discretization_multiseed_raw.csv", rows, per_seed_fields)

    summary = summarize_group(rows, "num_time_interval")
    n200_estimate_s = 200 * 6.2
    n400_estimate_s = 400 * 6.2
    for row in summary:
        row["n200_budget_estimate_s"] = n200_estimate_s
        row["n400_budget_estimate_s"] = n400_estimate_s
        row["n200_decision"] = "not_run_full_setting_estimated_about_20_7_min_per_seed"
        row["n400_decision"] = "skipped_estimated_about_41_3_min_per_seed"
    summary_fields = [
        "num_time_interval", "num_runs", "analytic_price", "mean_y0", "std_y0",
        "mean_relative_error_pct", "std_relative_error_pct", "mean_elapsed_time_s",
        "std_elapsed_time_s", "n200_budget_estimate_s", "n400_budget_estimate_s",
        "n200_decision", "n400_decision",
    ]
    write_csv(exp_dir / "time_discretization_multiseed_summary.csv", summary, summary_fields)
    draw_errorbar_chart(
        summary,
        "Time Discretization: Relative Error vs N",
        "n=3, low reliability; error bars show seed standard deviation",
        OUT_DIR / "experiment2_time_discretization" / "relative_error_vs_N.png",
        "num_time_interval",
        "mean_relative_error_pct",
        "std_relative_error_pct",
        "N",
        "Relative error (%)",
        log_x=True,
        log_y=False,
    )
    draw_errorbar_chart(
        summary,
        "Time Discretization: Training Time vs N",
        "n=3, low reliability; error bars show runtime standard deviation",
        OUT_DIR / "experiment2_time_discretization" / "training_time_vs_N.png",
        "num_time_interval",
        "mean_elapsed_time_s",
        "std_elapsed_time_s",
        "N",
        "Training time (s)",
        log_x=True,
        log_y=False,
    )
    return rows, summary


def align_histories(histories):
    shared_steps = sorted(set.intersection(*[set(item["step"] for item in history) for history in histories]))
    aligned = []
    for step in shared_steps:
        values = []
        for history in histories:
            row = next(item for item in history if item["step"] == step)
            values.append(row["y0"])
        mean, sd = stats(values)
        aligned.append((step, mean, mean - sd, mean + sd))
    return aligned


def experiment_batch_sensitivity_rerun():
    exp_dir = OUT_DIR / "experiment3_batch_optimization" / "rerun"
    base = load_json(BASE_BASKET_CONFIG)
    seeds = [4101, 4201, 4202]
    rows = []
    series = []
    analytic = None

    configs = [
        ("64", {"batch_size": 64}),
        ("128", {"batch_size": 128}),
        ("256", {"batch_size": 256}),
        ("512", {"batch_size": 512}),
        ("64_long", {"batch_size": 64, "num_iterations": 12000, "lr_boundaries": [4000, 8000]}),
    ]

    for label, net_updates in configs:
        histories = []
        batch_size = int(label.split("_")[0])
        max_iterations = net_updates.get("num_iterations", base["net_config"]["num_iterations"])
        for seed in seeds:
            updates = dict(net_updates)
            updates["logging_frequency"] = 25
            config = make_config(base, net_updates=updates)
            run_name = f"basket100d_batch_{label}_seed_{seed}"
            print(f"running {run_name}")
            history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
            histories.append(history)
            row = summarize_final(
                "100D Geometric Basket",
                history,
                analytic,
                {"batch_size": batch_size, "label": label, "seed": seed, "max_iterations": max_iterations},
            )
            row["first_step_rel_error_lt_0_5_pct"] = first_step_under(history, analytic, 0.5)
            rows.append(row)
        series.append({"label": f"batch {label}", "points": align_histories(histories)})

    per_seed_fields = [
        "label", "batch_size", "seed", "max_iterations", "first_step_rel_error_lt_0_5_pct",
        "name", "analytic_price", "final_y0", "final_loss", "absolute_error",
        "relative_error_pct", "elapsed_time_s",
    ]
    write_csv(exp_dir / "batch_optimization_multiseed_raw.csv", rows, per_seed_fields)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    summary = []
    for label, items in grouped.items():
        y0_mean, y0_std = stats([item["final_y0"] for item in items])
        err_mean, err_std = stats([item["relative_error_pct"] for item in items])
        time_mean, time_std = stats([item["elapsed_time_s"] for item in items])
        first_steps = [item["first_step_rel_error_lt_0_5_pct"] for item in items if item["first_step_rel_error_lt_0_5_pct"] != ""]
        first_mean, first_std = stats(first_steps) if first_steps else ("", "")
        summary.append({
            "label": label,
            "batch_size": items[0]["batch_size"],
            "max_iterations": items[0]["max_iterations"],
            "num_runs": len(items),
            "analytic_price": items[0]["analytic_price"],
            "mean_y0": y0_mean,
            "std_y0": y0_std,
            "mean_relative_error_pct": err_mean,
            "std_relative_error_pct": err_std,
            "mean_first_step_rel_error_lt_0_5_pct": first_mean,
            "std_first_step_rel_error_lt_0_5_pct": first_std,
            "mean_elapsed_time_s": time_mean,
            "std_elapsed_time_s": time_std,
        })
    summary = sorted(summary, key=lambda row: (float(row["max_iterations"]), float(row["batch_size"]), row["label"]))
    summary_fields = [
        "label", "batch_size", "max_iterations", "num_runs", "analytic_price",
        "mean_y0", "std_y0", "mean_relative_error_pct", "std_relative_error_pct",
        "mean_first_step_rel_error_lt_0_5_pct", "std_first_step_rel_error_lt_0_5_pct",
        "mean_elapsed_time_s", "std_elapsed_time_s",
    ]
    write_csv(exp_dir / "batch_optimization_multiseed_summary.csv", summary, summary_fields)
    draw_mean_band_chart(
        series,
        "Batch Size Sensitivity: Y0 Convergence",
        "Mean of three seeds; shaded band is +/- one standard deviation",
        OUT_DIR / "experiment3_batch_optimization" / "batch_size_y0_convergence.png",
        analytic=analytic,
    )
    return rows, summary


def generate_feedback_tables():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    time_raw = load_rows(OUT_DIR / "experiment2_time_discretization" / "rerun" / "time_discretization_multiseed_raw.csv")
    time_rows = []
    for n_value in [25, 50, 100]:
        rows_for_n = [row for row in time_raw if int(row["num_time_interval"]) == n_value]
        rows_for_n = sorted(rows_for_n, key=lambda row: int(row["seed"]))
        for seed_id, row in enumerate(rows_for_n):
            time_rows.append({
                "N": n_value,
                "seed_id": seed_id,
                "Y0": row["final_y0"],
                "rel_error_pct": row["relative_error_pct"],
                "elapsed_time_s": row["elapsed_time_s"],
            })
    write_csv(
        OUT_DIR / "experiment2_time_discretization" / "rerun" / "time_discretization_raw_3_values.csv",
        time_rows,
        ["N", "seed_id", "Y0", "rel_error_pct", "elapsed_time_s"],
    )
    write_csv(
        FEEDBACK_DIR / "task1_experiment2_raw_seed_values.csv",
        time_rows,
        ["N", "seed_id", "Y0", "rel_error_pct", "elapsed_time_s"],
    )

    base = load_json(BASE_BASKET_CONFIG)
    analytic = analytic_price(make_config(base))
    batch_raw = load_rows(OUT_DIR / "experiment3_batch_optimization" / "rerun" / "batch_optimization_multiseed_raw.csv")
    batch_rows = []
    for label in ["64", "128", "256", "512", "64_long"]:
        rows_for_label = [row for row in batch_raw if row["label"] == label]
        rows_for_label = sorted(rows_for_label, key=lambda row: int(row["seed"]))
        for seed_id, row in enumerate(rows_for_label):
            history_path = (
                OUT_DIR / "experiment3_batch_optimization" / "rerun" / "runs" /
                f"basket100d_batch_{label}_seed_{row['seed']}_training_history.csv"
            )
            history = read_history(history_path)
            batch_rows.append({
                "batch_size": label,
                "seed_id": seed_id,
                "first_hit_step": row["first_step_rel_error_lt_0_5_pct"],
                "settling_step": settling_step(history, analytic, 0.5) or "NaN",
                "final_rel_error_pct": row["relative_error_pct"],
            })
    write_csv(
        OUT_DIR / "experiment3_batch_optimization" / "rerun" / "batch_settling_time_raw.csv",
        batch_rows,
        ["batch_size", "seed_id", "first_hit_step", "settling_step", "final_rel_error_pct"],
    )
    write_csv(
        FEEDBACK_DIR / "task3_batch_settling_time_raw.csv",
        batch_rows,
        ["batch_size", "seed_id", "first_hit_step", "settling_step", "final_rel_error_pct"],
    )

    summary = []
    for label in ["64", "128", "256", "512", "64_long"]:
        items = [row for row in batch_rows if row["batch_size"] == label]
        first_mean, first_std = stats([row["first_hit_step"] for row in items if row["first_hit_step"] != ""])
        settling_values = [row["settling_step"] for row in items if row["settling_step"] != "NaN"]
        settling_mean, settling_std = stats(settling_values) if settling_values else ("", "")
        summary.append({
            "batch_size": label,
            "n": len(items),
            "first_hit_step_mean": first_mean,
            "first_hit_step_sd": first_std,
            "settling_step_mean": settling_mean,
            "settling_step_sd": settling_std,
            "num_not_settled": len([row for row in items if row["settling_step"] == "NaN"]),
        })
    write_csv(
        OUT_DIR / "experiment3_batch_optimization" / "rerun" / "batch_settling_time_summary.csv",
        summary,
        [
            "batch_size", "n", "first_hit_step_mean", "first_hit_step_sd",
            "settling_step_mean", "settling_step_sd", "num_not_settled",
        ],
    )
    write_csv(
        FEEDBACK_DIR / "task3_batch_settling_time_summary.csv",
        summary,
        [
            "batch_size", "n", "first_hit_step_mean", "first_hit_step_sd",
            "settling_step_mean", "settling_step_sd", "num_not_settled",
        ],
    )

    figure_note_rows = [
        {
            "figure": "Experiment 1 seed_variance_100d_y0.png",
            "status": "final-value scatter",
            "note": "The figure plots final Y0 points across seeds, not full training trajectories.",
        },
        {
            "figure": "Experiment 1 seed_variance_1d_y0.png",
            "status": "final-value scatter",
            "note": "The figure plots final Y0 points across seeds, not full training trajectories.",
        },
    ]
    write_csv(FEEDBACK_DIR / "task5_experiment1_figure_check.csv", figure_note_rows, ["figure", "status", "note"])
    return time_rows, batch_rows, summary


def experiment_precision_diagnostic():
    exp_dir = OUT_DIR / "experiment2_time_discretization" / "precision_diagnostic"
    base = load_json(BASE_BASKET_CONFIG)
    rows = []
    seed = 6101
    for num_steps in [25, 50, 100]:
        for dtype in ["float32", "float64"]:
            config = make_config(
                base,
                {"num_time_interval": num_steps, "sample_dtype": dtype},
                {"dtype": dtype},
            )
            run_name = f"basket100d_N_{num_steps}_{dtype}_seed_{seed}"
            print(f"running {run_name}")
            history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
            rows.append(summarize_final(
                "100D Geometric Basket",
                history,
                analytic,
                {"N": num_steps, "dtype": dtype, "seed": seed},
            ))
    output_rows = [{
        "N": row["N"],
        "dtype": row["dtype"],
        "Y0": row["final_y0"],
        "rel_error_pct": row["relative_error_pct"],
        "elapsed_time_s": row["elapsed_time_s"],
    } for row in rows]
    write_csv(
        exp_dir / "precision_diagnostic_float32_float64.csv",
        output_rows,
        ["N", "dtype", "Y0", "rel_error_pct", "elapsed_time_s"],
    )
    write_csv(
        FEEDBACK_DIR / "task2_precision_diagnostic_float32_float64.csv",
        output_rows,
        ["N", "dtype", "Y0", "rel_error_pct", "elapsed_time_s"],
    )
    return output_rows


def experiment_correlation_rerun_feedback():
    exp_dir = OUT_DIR / "experiment4_correlation" / "rerun"
    base = load_json(BASE_BASKET_CONFIG)
    seeds = [5101, 5201, 5202]
    rows = []
    derived_rows = []
    series = {}
    for rho in [0.0, 0.3, 0.5]:
        histories = []
        for seed_id, seed in enumerate(seeds):
            eqn_updates = {"sample_dtype": "float64"}
            if rho == 0.0:
                eqn_updates["eqn_name"] = "GeometricBasket100D"
            else:
                eqn_updates.update({
                    "eqn_name": "GeometricBasket100DCorrelated",
                    "rho": rho,
                    "_comment": f"100D geometric basket with equicorrelation rho={rho}.",
                })
            config = make_config(base, eqn_updates, {"logging_frequency": 25})
            expected = analytic_price(config)
            config["net_config"]["y_init_range"] = [max(0.0, expected - 3.0), expected + 3.0]
            run_name = f"basket100d_rho_{str(rho).replace('.', 'p')}_seed_{seed}"
            print(f"running {run_name}")
            history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
            histories.append(history)
            final = summarize_final(f"rho={rho}", history, analytic, {"rho": rho, "seed_id": seed_id, "seed": seed})
            rows.append({
                "rho": rho,
                "seed_id": seed_id,
                "Y0": final["final_y0"],
                "rel_error_pct": final["relative_error_pct"],
                "elapsed_time_s": final["elapsed_time_s"],
            })
            derived_rows.append({
                "rho": rho,
                "seed_id": seed_id,
                "y0_std_steps_1000_1500": y0_std_in_window(history, 1000, 1500),
                "settling_step": settling_step(history, analytic, 0.5) or "NaN",
            })
        series[f"rho={rho}"] = align_histories(histories)
    write_csv(
        exp_dir / "correlation_multiseed_raw.csv",
        rows,
        ["rho", "seed_id", "Y0", "rel_error_pct", "elapsed_time_s"],
    )
    write_csv(
        exp_dir / "correlation_multiseed_diagnostics.csv",
        derived_rows,
        ["rho", "seed_id", "y0_std_steps_1000_1500", "settling_step"],
    )
    write_csv(
        FEEDBACK_DIR / "task4_correlation_multiseed_raw.csv",
        rows,
        ["rho", "seed_id", "Y0", "rel_error_pct", "elapsed_time_s"],
    )
    write_csv(
        FEEDBACK_DIR / "task4_correlation_multiseed_diagnostics.csv",
        derived_rows,
        ["rho", "seed_id", "y0_std_steps_1000_1500", "settling_step"],
    )
    draw_mean_band_chart(
        [{"label": label, "points": points} for label, points in series.items()],
        "Correlated Basket: Y0 Convergence by rho",
        "Mean of three seeds; shaded band is +/- one standard deviation",
        OUT_DIR / "experiment4_correlation" / "rho_multiseed_y0_convergence.png",
        analytic=None,
    )
    return rows, derived_rows


def experiment_correlation():
    exp_dir = OUT_DIR / "experiment4_correlation"
    base = load_json(BASE_BASKET_CONFIG)
    rows = []
    seed = 5101
    series = []

    independent_history = read_history(ROOT / "logs" / "geometric_basket_100d_training_history.csv")
    independent_config = make_config(base)
    independent_analytic = analytic_price(independent_config)
    rows.append(summarize_final("rho=0.0", independent_history, independent_analytic, {"rho": 0.0, "seed": "phase1"}))

    for rho in [0.3, 0.5]:
        config = make_config(base, {"eqn_name": "GeometricBasket100DCorrelated", "rho": rho, "_comment": f"100D geometric basket with equicorrelation rho={rho}."})
        # Correlation raises the analytic price, so widen the learned Y0 range.
        expected = analytic_price(config)
        config["net_config"]["y_init_range"] = [max(0.0, expected - 3.0), expected + 3.0]
        run_name = f"basket100d_rho_{str(rho).replace('.', 'p')}"
        print(f"running {run_name}")
        history, analytic = run_training(config, exp_dir / "runs", run_name, seed)
        rows.append(summarize_final(f"rho={rho}", history, analytic, {"rho": rho, "seed": seed}))
        draw_line_chart(
            [{"label": f"rho={rho}", "points": [(item["step"], item["y0"]) for item in history]}],
            f"Correlated Basket: rho={rho}",
            "Y0 convergence with analytic correlated geometric-basket price",
            exp_dir / f"rho_{str(rho).replace('.', 'p')}_y0_convergence.png",
            analytic=analytic,
        )
        series.append({"label": f"rho={rho}", "points": [(item["step"], item["y0"]) for item in history]})

    fieldnames = ["rho", "seed", "name", "analytic_price", "final_y0", "final_loss", "absolute_error", "relative_error_pct", "elapsed_time_s"]
    write_csv(exp_dir / "correlation_error_comparison.csv", rows, fieldnames)
    return rows


def load_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def stats(values):
    arr = np.array([float(value) for value in values], dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def markdown_table(rows, columns):
    header = "| " + " | ".join(label for label, _ in columns) + " |\n"
    divider = "|" + "|".join("---" if idx == 0 else "---:" for idx in range(len(columns))) + "|\n"
    body = ""
    for row in rows:
        body += "| " + " | ".join(formatter(row) for _, formatter in columns) + " |\n"
    return header + divider + body


def pm(mean_value, std_value, digits=4):
    return f"{float(mean_value):.{digits}f} +/- {float(std_value):.{digits}f}"


def generate_report():
    exp1 = OUT_DIR / "experiment1_seed_variance"
    exp2 = OUT_DIR / "experiment2_time_discretization"
    exp3 = OUT_DIR / "experiment3_batch_optimization"
    exp4 = OUT_DIR / "experiment4_correlation"

    seed_rows = load_rows(exp1 / "seed_variance_100d.csv")
    seed_1d_rows = load_rows(exp1 / "seed_variance_1d.csv")
    time_rows = load_rows(exp2 / "time_discretization.csv")
    time_rerun_raw_rows = load_rows(exp2 / "rerun" / "time_discretization_multiseed_raw.csv")
    time_rerun_rows = load_rows(exp2 / "rerun" / "time_discretization_multiseed_summary.csv")
    batch_rows = load_rows(exp3 / "batch_optimization_sensitivity.csv")
    batch_rerun_rows = load_rows(exp3 / "rerun" / "batch_optimization_multiseed_summary.csv")
    corr_rows = load_rows(exp4 / "correlation_error_comparison.csv")
    task2_rows = load_rows(exp2 / "precision_diagnostic" / "precision_diagnostic_float32_float64.csv")
    task3_raw_rows = load_rows(exp3 / "rerun" / "batch_settling_time_raw.csv")
    task4_raw_rows = load_rows(exp4 / "rerun" / "correlation_multiseed_raw.csv")
    task4_diag_rows = load_rows(exp4 / "rerun" / "correlation_multiseed_diagnostics.csv")

    report_path = ROOT / "progress_report_phase2_error_analysis.md"
    y0_mean, y0_std = stats([row["final_y0"] for row in seed_rows]) if seed_rows else (0.0, 0.0)
    err_mean, err_std = stats([row["relative_error_pct"] for row in seed_rows]) if seed_rows else (0.0, 0.0)
    y0_1d_mean, y0_1d_std = stats([row["final_y0"] for row in seed_1d_rows]) if seed_1d_rows else (0.0, 0.0)
    err_1d_mean, err_1d_std = stats([row["relative_error_pct"] for row in seed_1d_rows]) if seed_1d_rows else (0.0, 0.0)

    time_table = markdown_table(time_rows, [
        ("N", lambda r: r["num_time_interval"]),
        ("Status", lambda r: r.get("status", "completed")),
        ("Y0", lambda r: f"{float(r['final_y0']):.6f}" if r.get("final_y0") else ""),
        ("Rel. Error (%)", lambda r: f"{float(r['relative_error_pct']):.4f}" if r.get("relative_error_pct") else ""),
        ("Time", lambda r: f"{float(r['elapsed_time_s']):.1f}s" if r.get("elapsed_time_s") else ""),
    ]) if time_rows else "_Not completed yet._\n"

    time_range_rows = []
    for n_value in sorted({int(row["num_time_interval"]) for row in time_rerun_raw_rows}):
        group = [row for row in time_rerun_raw_rows if int(row["num_time_interval"]) == n_value]
        y0_values = [float(row["final_y0"]) for row in group]
        error_values = [float(row["relative_error_pct"]) for row in group]
        time_values = [float(row["elapsed_time_s"]) for row in group]
        time_range_rows.append({
            "N": n_value,
            "runs": len(group),
            "y0_min": min(y0_values),
            "y0_max": max(y0_values),
            "error_min": min(error_values),
            "error_max": max(error_values),
            "time_min": min(time_values),
            "time_max": max(time_values),
        })

    time_rerun_table = markdown_table(time_range_rows, [
        ("N", lambda r: str(r["N"])),
        ("Runs", lambda r: str(r["runs"])),
        ("Y0 [min, max]", lambda r: f"[{r['y0_min']:.6f}, {r['y0_max']:.6f}]"),
        ("Rel. Error (%) [min, max]", lambda r: f"[{r['error_min']:.4f}, {r['error_max']:.4f}]"),
        ("Time (s) [min, max]", lambda r: f"[{r['time_min']:.1f}, {r['time_max']:.1f}]"),
    ]) if time_range_rows else "_Supplemental rerun not completed yet._\n"

    if time_rerun_raw_rows:
        draw_scatter(
            time_rerun_raw_rows,
            "Time Discretization: Relative Error vs N",
            "Individual results from three seeds at each N",
            exp2 / "relative_error_vs_N.png",
            "num_time_interval",
            "relative_error_pct",
            x_label="N",
        )
        draw_scatter(
            time_rerun_raw_rows,
            "Time Discretization: Training Time vs N",
            "Individual results from three seeds at each N",
            exp2 / "training_time_vs_N.png",
            "num_time_interval",
            "elapsed_time_s",
            x_label="N",
        )

    batch_table = markdown_table(batch_rows, [
        ("Batch", lambda r: r["batch_size"]),
        ("Max Iter.", lambda r: r["max_iterations"]),
        ("First <0.5%", lambda r: r["first_step_rel_error_lt_0_5_pct"] or "not reached"),
        ("Final Rel. Error (%)", lambda r: f"{float(r['relative_error_pct']):.4f}"),
        ("Time", lambda r: f"{float(r['elapsed_time_s']):.1f}s"),
    ]) if batch_rows else "_Not completed yet._\n"

    corr_table = markdown_table(corr_rows, [
        ("rho", lambda r: str(r["rho"])),
        ("Analytic", lambda r: f"{float(r['analytic_price']):.6f}"),
        ("Y0", lambda r: f"{float(r['final_y0']):.6f}"),
        ("Abs. Error", lambda r: f"{float(r['absolute_error']):.6f}"),
        ("Rel. Error (%)", lambda r: f"{float(r['relative_error_pct']):.4f}"),
    ]) if corr_rows else "_Not completed yet._\n"

    task1_table = markdown_table(time_rerun_raw_rows, [
        ("N", lambda r: r["num_time_interval"]),
        ("Seed", lambda r: r["seed"]),
        ("Y0", lambda r: f"{float(r['final_y0']):.6f}"),
        ("Relative Error (%)", lambda r: f"{float(r['relative_error_pct']):.4f}"),
        ("Time (s)", lambda r: f"{float(r['elapsed_time_s']):.1f}"),
    ]) if time_rerun_raw_rows else "_Not generated yet._\n"

    task2_table = markdown_table(task2_rows, [
        ("N", lambda r: r["N"]),
        ("Precision", lambda r: r["dtype"]),
        ("Y0", lambda r: f"{float(r['Y0']):.6f}"),
        ("Relative Error (%)", lambda r: f"{float(r['rel_error_pct']):.4f}"),
        ("Time (s)", lambda r: f"{float(r['elapsed_time_s']):.1f}"),
    ]) if task2_rows else "_Not generated yet._\n"

    precision_pair_rows = []
    for n_value in sorted({int(row["N"]) for row in task2_rows}):
        group = {row["dtype"]: row for row in task2_rows if int(row["N"]) == n_value}
        if "float32" not in group or "float64" not in group:
            continue
        row32, row64 = group["float32"], group["float64"]
        precision_pair_rows.append({
            "N": n_value,
            "delta_y0": abs(float(row32["Y0"]) - float(row64["Y0"])),
            "delta_error": abs(float(row32["rel_error_pct"]) - float(row64["rel_error_pct"])),
            "time_reduction": 100.0 * (
                1.0 - float(row32["elapsed_time_s"]) / float(row64["elapsed_time_s"])
            ),
        })

    precision_pair_table = markdown_table(precision_pair_rows, [
        ("N", lambda r: str(r["N"])),
        ("Abs. difference in Y0", lambda r: f"{r['delta_y0']:.6f}"),
        ("Abs. difference in rel. error (pp)", lambda r: f"{r['delta_error']:.4f}"),
        ("Float32 time reduction", lambda r: f"{r['time_reduction']:.1f}%"),
    ]) if precision_pair_rows else "_Paired comparison not available._\n"

    task3_raw_table = markdown_table(task3_raw_rows, [
        ("Batch", lambda r: r["batch_size"]),
        ("Run", lambda r: r["seed_id"]),
        ("Settling Step", lambda r: "not settled" if r["settling_step"] == "NaN" else r["settling_step"]),
        ("Final Relative Error (%)", lambda r: f"{float(r['final_rel_error_pct']):.4f}"),
    ]) if task3_raw_rows else "_Not generated yet._\n"

    settling_range_rows = []
    batch_order = ["64", "128", "256", "512", "64_long"]
    for label in batch_order:
        group = sorted(
            [row for row in task3_raw_rows if row["batch_size"] == label],
            key=lambda row: int(row["seed_id"]),
        )
        if not group:
            continue
        settling_values = [
            int(row["settling_step"]) for row in group if row["settling_step"] != "NaN"
        ]
        final_errors = [float(row["final_rel_error_pct"]) for row in group]
        settling_range_rows.append({
            "batch_size": label,
            "runs": len(group),
            "max_iterations": 12000 if label == "64_long" else 6000,
            "settling_by_seed": ", ".join(
                "not settled" if row["settling_step"] == "NaN" else row["settling_step"]
                for row in group
            ),
            "settling_min": min(settling_values) if settling_values else None,
            "settling_max": max(settling_values) if settling_values else None,
            "num_not_settled": sum(row["settling_step"] == "NaN" for row in group),
            "error_min": min(final_errors),
            "error_max": max(final_errors),
        })

    task3_summary_table = markdown_table(settling_range_rows, [
        ("Batch", lambda r: r["batch_size"]),
        ("Runs", lambda r: str(r["runs"])),
        ("Max Iter.", lambda r: str(r["max_iterations"])),
        ("Settling steps (seed 0, 1, 2)", lambda r: r["settling_by_seed"]),
        ("Settled [min, max]", lambda r: (
            f"[{r['settling_min']}, {r['settling_max']}]"
            if r["settling_min"] is not None else "none"
        )),
        ("Not settled", lambda r: str(r["num_not_settled"])),
        ("Final rel. error (%) [min, max]", lambda r: f"[{r['error_min']:.4f}, {r['error_max']:.4f}]"),
    ]) if settling_range_rows else "_Not generated yet._\n"

    batch_rerun_table = task3_summary_table

    task4_raw_table = markdown_table(task4_raw_rows, [
        ("rho", lambda r: r["rho"]),
        ("Run", lambda r: r["seed_id"]),
        ("Y0", lambda r: f"{float(r['Y0']):.6f}"),
        ("Relative Error (%)", lambda r: f"{float(r['rel_error_pct']):.4f}"),
        ("Time (s)", lambda r: f"{float(r['elapsed_time_s']):.1f}"),
    ]) if task4_raw_rows else "_Not generated yet._\n"

    task4_diag_table = markdown_table(task4_diag_rows, [
        ("rho", lambda r: r["rho"]),
        ("Run", lambda r: r["seed_id"]),
        ("Y0 SD, Steps 1000-1500", lambda r: f"{float(r['y0_std_steps_1000_1500']):.6f}"),
        ("Settling Step", lambda r: r["settling_step"]),
    ]) if task4_diag_rows else "_Not generated yet._\n"

    corr_analytic = {float(row["rho"]): float(row["analytic_price"]) for row in corr_rows}
    correlation_range_rows = []
    for rho_value in sorted({float(row["rho"]) for row in task4_raw_rows}):
        group = [row for row in task4_raw_rows if float(row["rho"]) == rho_value]
        y0_values = [float(row["Y0"]) for row in group]
        error_values = [float(row["rel_error_pct"]) for row in group]
        time_values = [float(row["elapsed_time_s"]) for row in group]
        correlation_range_rows.append({
            "rho": rho_value,
            "runs": len(group),
            "analytic": corr_analytic.get(rho_value, float("nan")),
            "y0_min": min(y0_values),
            "y0_max": max(y0_values),
            "error_min": min(error_values),
            "error_max": max(error_values),
            "time_min": min(time_values),
            "time_max": max(time_values),
        })

    correlation_range_table = markdown_table(correlation_range_rows, [
        ("rho", lambda r: f"{r['rho']:.1f}"),
        ("Runs", lambda r: str(r["runs"])),
        ("Analytic", lambda r: f"{r['analytic']:.6f}"),
        ("Y0 [min, max]", lambda r: f"[{r['y0_min']:.5f}, {r['y0_max']:.5f}]"),
        ("Rel. Error (%) [min, max]", lambda r: f"[{r['error_min']:.4f}, {r['error_max']:.4f}]"),
        ("Time (s) [min, max]", lambda r: f"[{r['time_min']:.1f}, {r['time_max']:.1f}]"),
    ]) if correlation_range_rows else "_Three-seed rerun not completed yet._\n"

    if task4_raw_rows:
        draw_scatter(
            task4_raw_rows,
            "Correlated Basket: Relative Error by rho",
            "Individual results from three seeds at each rho",
            exp4 / "rho_multiseed_relative_error.png",
            "rho",
            "rel_error_pct",
            x_label="rho",
        )

    attribution_rows = [
        {
            "source": "Random seed / sampling noise",
            "metric": "100D final Y0 std across seeds",
            "magnitude": f"{y0_std:.6f} Y0 units; rel. error std {err_std:.4f}%",
        },
        {
            "source": "Low-dimensional comparison",
            "metric": "1D final Y0 std across seeds",
            "magnitude": f"{y0_1d_std:.6f} Y0 units; rel. error std {err_1d_std:.4f}%",
        },
        {
            "source": "Time discretization",
            "metric": "Observed rel. error range across three seeds",
            "magnitude": "; ".join(
                f"N={row['N']}: [{row['error_min']:.4f}, {row['error_max']:.4f}]%"
                for row in time_range_rows
            ) if time_range_rows else "not completed",
        },
        {
            "source": "Network optimization",
            "metric": "Settling steps and final-error ranges across three seeds",
            "magnitude": "; ".join(
                f"batch {row['batch_size']}: settling {row['settling_by_seed']}; "
                f"error [{row['error_min']:.4f}, {row['error_max']:.4f}]%"
                for row in settling_range_rows
                if row["max_iterations"] == 6000
            ) if settling_range_rows else "not completed",
        },
        {
            "source": "Correlation extension",
            "metric": "Observed rel. error range across three seeds",
            "magnitude": "; ".join(
                f"rho={row['rho']:.1f}: [{row['error_min']:.4f}, {row['error_max']:.4f}]%"
                for row in correlation_range_rows
            ) if correlation_range_rows else "not completed",
        },
    ]
    attribution_table = markdown_table(attribution_rows, [
        ("Error Source", lambda r: r["source"]),
        ("Diagnostic", lambda r: r["metric"]),
        ("Observed Magnitude", lambda r: r["magnitude"]),
    ])

    report = rf"""# Phase 2 Experimental Report: Error Source Analysis and Robustness of Deep BSDE Pricing

## Objective

This phase quantifies the sensitivity of the Deep BSDE geometric-basket pricer to random initialization and sampling, time discretization, numerical precision, network batch size, and cross-asset correlation.

## Common Experimental Setup

The principal benchmark is a 100-dimensional geometric-average European call with

| Parameter | Value |
|---|---:|
| Dimension \(d\) | 100 |
| Maturity \(T\) | 1 year |
| Initial asset prices \(S_0^i\) | 100 |
| Risk-free rate \(r\) | 0.05 |
| Volatility \(\sigma\) | 0.20 |
| Strike \(K\) | 100 |
| Baseline time intervals | 20 |
| Training iterations | 6000 unless stated otherwise |
| Baseline batch size | 64 |
| Hidden-layer widths | [110, 110] |
| Learning rates | \(5\times10^{{-3}},\,2\times10^{{-3}},\,10^{{-3}}\) |
| Learning-rate boundaries | 2000 and 4000 steps |
| Validation sample size | 512 |
| Baseline numerical precision | float64 |

For each completed run, the reported estimate is the final learned value of \(Y_0\).  Accuracy is measured against the analytic geometric-basket price using

$$
\text{{relative error}}(\%)=\frac{{|Y_0-Y_0^{{\mathrm{{analytic}}}}|}}{{Y_0^{{\mathrm{{analytic}}}}}}\times100.
$$

The analytic value for the independent 100D baseline is \(Y_0^{{\mathrm{{analytic}}}}=2.971854\).

## Experiment 1: Random-Seed Sensitivity

### Design

The full 100D model was trained independently with eight seeds (`1001`-`1008`).  A one-dimensional Black-Scholes model was also trained with five seeds (`2001`-`2005`) as a low-dimensional comparison.  Every point in the figures is the final \(Y_0\) from one complete training run.

### Results

For the 100D geometric basket (\(n=8\)), the final \(Y_0\) mean is `{y0_mean:.6f}` with standard deviation `{y0_std:.6f}`.  The relative-error mean is `{err_mean:.4f}%` with standard deviation `{err_std:.4f}%`.

For the 1D comparison (\(n=5\)), the final \(Y_0\) mean is `{y0_1d_mean:.6f}` with standard deviation `{y0_1d_std:.6f}`.  The relative-error mean is `{err_1d_mean:.4f}%` with standard deviation `{err_1d_std:.4f}%`.

![100D final Y0 across eight seeds](reports/phase2_error_analysis/experiment1_seed_variance/seed_variance_100d_y0.png)

![1D final Y0 across five seeds](reports/phase2_error_analysis/experiment1_seed_variance/seed_variance_1d_y0.png)

The larger variation in the 100D problem indicates that sampling and optimization variability become more pronounced with dimension.  Raw results are stored in `reports/phase2_error_analysis/experiment1_seed_variance/seed_variance_100d.csv` and `reports/phase2_error_analysis/experiment1_seed_variance/seed_variance_1d.csv`.

## Experiment 2: Time-Discretization Sensitivity

### Design

The number of time intervals was set to \(N=25,50,100\).  Each setting used the same three seeds (`3101`, `3201`, and `3202`), with all other model and training settings fixed.  Because each condition has three runs, the individual observations and observed ranges are reported directly.

### Per-run Results

{task1_table}
### Observed Ranges

{time_rerun_table}
![Relative error versus number of time intervals](reports/phase2_error_analysis/experiment2_time_discretization/relative_error_vs_N.png)

![Training time versus number of time intervals](reports/phase2_error_analysis/experiment2_time_discretization/training_time_vs_N.png)

The relative-error ranges overlap across all three values of \(N\); these runs therefore do not establish a monotone accuracy improvement as the time grid is refined.  Runtime increases substantially with \(N\), from 145-163 seconds at \(N=25\) to 584-607 seconds at \(N=100\).  The tested range supports conclusions only for \(N=25,50,100\).

Raw data are stored in `reports/phase2_error_analysis/experiment2_time_discretization/rerun/time_discretization_multiseed_raw.csv`.

### Numerical-Precision Diagnostic

For each \(N\), one float32 run and one float64 run used seed `6101` and identical time-discretization and network settings.  This paired design reduces sampling differences and isolates the effect of numerical precision more directly, although dtype-dependent changes in the optimization trajectory remain possible.

{task2_table}
Paired differences are

{precision_pair_table}
The largest observed change in \(Y_0\) is 0.000710, and the largest change in relative error is 0.0222 percentage points.  Float32 reduces runtime by 18.7%-36.0% in these pairs.  No systematic accuracy advantage for float64 is visible in this limited diagnostic.  A formal roundoff-error decomposition would require an extended-precision reference and more paired repetitions.

Raw data are stored in `reports/phase2_error_analysis/experiment2_time_discretization/precision_diagnostic/precision_diagnostic_float32_float64.csv`.

## Experiment 3: Batch-Size and Optimization Sensitivity

### Design and Convergence Metric

Batch sizes 64, 128, 256, and 512 were trained for 6000 iterations with seeds `4101`, `4201`, and `4202` in each setting.  In the tables, these seeds are indexed as runs 0, 1, and 2.  An additional batch-64 condition was trained for 12000 iterations.  \(Y_0\) was recorded every 25 steps.

The settling step is defined as the earliest recorded step \(s\) for which

$$
\frac{{|Y_0^{{(k)}}-Y_0^{{\mathrm{{analytic}}}}|}}{{Y_0^{{\mathrm{{analytic}}}}}}\times100\leq0.5\%
\qquad\text{{for every recorded }}k\geq s.
$$

This metric measures persistent convergence through the end of the specified training horizon.

### Per-run Results

{task3_raw_table}
### Observed Ranges

{task3_summary_table}
Among the 6000-step conditions, batch 512 has the earliest upper endpoint of the observed settling range (`3950`), while batch 64 settles at steps `5550` and `5750` in two runs and does not settle in the third.  The batch-256 results show substantial run-to-run variation (`1375`-`5200`).  The 12000-step batch-64 condition is not directly comparable to the 6000-step conditions because persistence is evaluated over a longer terminal horizon.

Raw data are stored in `reports/phase2_error_analysis/experiment3_batch_optimization/rerun/batch_settling_time_raw.csv`.

## Experiment 4: Correlated Geometric-Basket Extension

### Model

Here `rho` is the common pairwise instantaneous correlation between the Brownian shocks driving different assets.  For `d=100`,

$$
dS_t^i = rS_t^i\,dt + \sigma S_t^i\,dW_t^i, \qquad i=1,\ldots,d,
$$

with

$$
d\langle W^i,W^j\rangle_t = \rho_{{ij}}\,dt,
\qquad
\rho_{{ij}}=
\begin{{cases}}
1, & i=j,\\
\rho, & i\ne j.
\end{{cases}}
$$

The geometric-basket terminal payoff and the BSDE integrated by the solver are

$$
G_T=\left(\prod_{{i=1}}^d S_T^i\right)^{{1/d}},
\qquad
Y_T=(G_T-K)^+,
$$

$$
dY_t=rY_t\,dt+Z_t^\top dW_t.
$$

Equivalently, writing `Y_t=u(t,S_t)`, the corresponding pricing PDE is

$$
\frac{{\partial u}}{{\partial t}}
+r\sum_{{i=1}}^d s_i\frac{{\partial u}}{{\partial s_i}}
+\frac{{\sigma^2}}2\sum_{{i=1}}^d\sum_{{j=1}}^d
\rho_{{ij}}s_is_j\frac{{\partial^2u}}{{\partial s_i\partial s_j}}
-ru=0,
$$

with terminal condition

$$
u(T,s)=\left[\left(\prod_{{i=1}}^d s_i\right)^{{1/d}}-K\right]^+.
$$

The experiments use `rho = 0, 0.3, 0.5`; `rho = 0` gives independent shocks, while positive `rho` reduces diversification and increases the effective volatility of the geometric basket.

### Design

Each value of `rho` used the same three seeds (`5101`, `5201`, and `5202`) and identical settings, including 25-step logging.  Individual results and observed ranges are reported.

### Per-run Results

{task4_raw_table}
### Observed Ranges

{correlation_range_table}
![Correlation relative error by rho](reports/phase2_error_analysis/experiment4_correlation/rho_multiseed_relative_error.png)

All nine estimates are within 0.3711% of the corresponding analytic value.  The analytic option value increases from 2.971854 at `rho=0` to 7.662222 at `rho=0.5`, consistent with the higher effective basket volatility under stronger positive correlation.

### Training-Stability Diagnostics

The temporal standard deviation is computed from logged \(Y_0\) values between steps 1000 and 1500.  The settling step uses the same persistent 0.5% error-band definition as Experiment 3.

{task4_diag_table}
Raw results are stored in `reports/phase2_error_analysis/experiment4_correlation/rerun/correlation_multiseed_raw.csv` and `reports/phase2_error_analysis/experiment4_correlation/rerun/correlation_multiseed_diagnostics.csv`.

## Error Attribution Summary

{attribution_table}

## Limitations

- Experiments 2-4 contain three runs per condition, so their comparisons are descriptive rather than precise estimates of population variability.
- The precision diagnostic contains one matched pair per value of \(N\); it cannot by itself separate all optimization effects from roundoff effects.
- Settling steps are resolved only to the 25-step logging interval.
- Time-discretization conclusions are limited to \(N=25,50,100\).
- The correlation experiment assumes one common pairwise correlation for every pair of assets.

## Conclusion

The 100D Deep BSDE estimate remains close to the analytic geometric-basket benchmark across the tested seeds, time grids, batch sizes, numerical precisions, and correlation levels.  Seed-to-seed and optimization variation are visible at the sub-percent scale.  Increasing \(N\) raises computational cost without producing a monotone accuracy improvement over the tested range.  Persistent convergence depends strongly on both batch size and seed, while the correlated model remains accurate for `rho=0.3` and `rho=0.5` under the three-run protocol.
"""
    report_path.write_text(report)
    print(f"wrote {report_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=[
            "seed", "time", "time_collect", "time_rerun", "batch", "batch_rerun",
            "corr", "feedback_tables", "precision_diag", "corr_rerun", "report", "all",
        ],
        help="Which Phase 2 experiment to run.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("seed", "all"):
        experiment_seed_variance()
    if args.experiment in ("time", "all"):
        experiment_time_discretization()
    if args.experiment == "time_collect":
        collect_time_discretization()
    if args.experiment in ("time_rerun",):
        experiment_time_discretization_rerun()
    if args.experiment in ("batch", "all"):
        experiment_batch_sensitivity()
    if args.experiment in ("batch_rerun",):
        experiment_batch_sensitivity_rerun()
    if args.experiment in ("corr", "all"):
        experiment_correlation()
    if args.experiment == "feedback_tables":
        generate_feedback_tables()
    if args.experiment == "precision_diag":
        experiment_precision_diagnostic()
    if args.experiment == "corr_rerun":
        experiment_correlation_rerun_feedback()
    if args.experiment in ("report", "all"):
        generate_report()


if __name__ == "__main__":
    main()
