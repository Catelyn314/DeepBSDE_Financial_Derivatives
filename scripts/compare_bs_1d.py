"""Compare the trained Deep BSDE Y0 with the analytic 1D Black-Scholes price."""

import argparse
import csv
import json
from math import erf, exp, log, sqrt


def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call_price(spot, strike, rate, sigma, maturity):
    d1 = (log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return spot * normal_cdf(d1) - strike * exp(-rate * maturity) * normal_cdf(d2)


def read_final_y0(history_path):
    with open(history_path, newline="") as history_file:
        rows = list(csv.DictReader(history_file))
    if not rows:
        raise ValueError("Training history is empty.")
    final_row = rows[-1]
    return {
        "step": int(final_row["step"]),
        "loss": float(final_row["loss_function"]),
        "model_y0": float(final_row["target_value"]),
        "elapsed_time": float(final_row["elapsed_time"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/bs_1d.json")
    parser.add_argument("--history_path", default="logs/bs_1d_training_history.csv")
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config = json.load(config_file)
    eqn = config["eqn_config"]

    analytic = black_scholes_call_price(
        spot=eqn["x_init"],
        strike=eqn["strike"],
        rate=eqn["rate"],
        sigma=eqn["sigma"],
        maturity=eqn["total_time"],
    )
    result = read_final_y0(args.history_path)
    abs_error = abs(result["model_y0"] - analytic)
    rel_error = abs_error / analytic * 100.0

    print("Black-Scholes 1D baseline")
    print(f"final step: {result['step']}")
    print(f"final loss: {result['loss']:.6e}")
    print(f"model Y0: {result['model_y0']:.6f}")
    print(f"analytic price: {analytic:.6f}")
    print(f"absolute error: {abs_error:.6f}")
    print(f"relative error (%): {rel_error:.4f}")
    print(f"elapsed time (s): {result['elapsed_time']:.1f}")


if __name__ == "__main__":
    main()
