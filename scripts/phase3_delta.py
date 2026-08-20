"""Common-random-number bump-and-revalue Delta engine for Phase 3."""

from dataclasses import asdict, dataclass
from math import erf, exp, log, sqrt
from time import perf_counter

import numpy as np


@dataclass
class DeltaEstimate:
    dim: int
    bump: float
    paths: int
    seed: int
    delta: np.ndarray
    standard_error: np.ndarray
    raw_standard_error: np.ndarray
    beta: np.ndarray
    price_plus: np.ndarray
    price_minus: np.ndarray
    baseline_price: float
    elapsed_seconds: float

    def metadata(self) -> dict:
        result = asdict(self)
        for key in (
            "delta",
            "standard_error",
            "raw_standard_error",
            "beta",
            "price_plus",
            "price_minus",
        ):
            result[key] = result[key].tolist()
        return result


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def geometric_call_price(
    geometric_spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    dim: int,
) -> float:
    variance = sigma * sigma * maturity / dim
    std = sqrt(variance)
    log_mean = (
        log(geometric_spot)
        + (rate - 0.5 * sigma * sigma) * maturity
    )
    d2 = (log_mean - log(strike)) / std
    d1 = d2 + std
    discounted_forward = exp(
        -rate * maturity + log_mean + 0.5 * variance
    )
    return (
        discounted_forward * normal_cdf(d1)
        - strike * exp(-rate * maturity) * normal_cdf(d2)
    )


def crn_bump_delta(
    dim: int,
    bump: float,
    paths: int,
    seed: int,
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.05,
    sigma: float = 0.2,
    maturity: float = 1.0,
    chunk_size: int = 10_000,
) -> DeltaEstimate:
    """Estimate all component Deltas using CRN and a geometric-payoff CV."""
    if not 0.0 < bump < spot:
        raise ValueError("bump must lie strictly between zero and spot.")
    if paths < 2:
        raise ValueError("At least two paths are required.")

    started = perf_counter()
    rng = np.random.default_rng(seed)
    discount = exp(-rate * maturity)
    drift = (rate - 0.5 * sigma * sigma) * maturity
    diffusion_scale = sigma * sqrt(maturity)
    geometric_plus_multiplier = ((spot + bump) / spot) ** (1.0 / dim)
    geometric_minus_multiplier = ((spot - bump) / spot) ** (1.0 / dim)
    known_plus = geometric_call_price(
        spot * geometric_plus_multiplier,
        strike,
        rate,
        sigma,
        maturity,
        dim,
    )
    known_minus = geometric_call_price(
        spot * geometric_minus_multiplier,
        strike,
        rate,
        sigma,
        maturity,
        dim,
    )

    count = 0
    mean_delta_y = np.zeros(dim, dtype=np.float64)
    mean_delta_x = 0.0
    m2_delta_y = np.zeros(dim, dtype=np.float64)
    m2_delta_x = 0.0
    covariance = np.zeros(dim, dtype=np.float64)
    sum_y_plus = np.zeros(dim, dtype=np.float64)
    sum_y_minus = np.zeros(dim, dtype=np.float64)
    sum_x_plus = 0.0
    sum_x_minus = 0.0
    baseline_count = 0
    baseline_mean_y = 0.0
    baseline_mean_x = 0.0
    baseline_m2_x = 0.0
    baseline_covariance = 0.0

    remaining = paths
    while remaining:
        current = min(chunk_size, remaining)
        shocks = rng.standard_normal(
            size=(current, dim),
            dtype=np.float64,
        )
        factors = np.exp(drift + diffusion_scale * shocks)
        arithmetic_base = spot * np.mean(factors, axis=1, dtype=np.float64)
        arithmetic_plus = arithmetic_base[:, None] + bump * factors / dim
        arithmetic_minus = arithmetic_base[:, None] - bump * factors / dim
        y_plus = discount * np.maximum(arithmetic_plus - strike, 0.0)
        y_minus = discount * np.maximum(arithmetic_minus - strike, 0.0)

        geometric_base = spot * np.exp(
            np.mean(np.log(factors), axis=1, dtype=np.float64)
        )
        y_base = discount * np.maximum(arithmetic_base - strike, 0.0)
        x_base = discount * np.maximum(geometric_base - strike, 0.0)
        x_plus = discount * np.maximum(
            geometric_base * geometric_plus_multiplier - strike,
            0.0,
        )
        x_minus = discount * np.maximum(
            geometric_base * geometric_minus_multiplier - strike,
            0.0,
        )
        delta_y = (y_plus - y_minus) / (2.0 * bump)
        delta_x = (x_plus - x_minus) / (2.0 * bump)

        batch_mean_y = np.mean(delta_y, axis=0, dtype=np.float64)
        batch_mean_x = float(np.mean(delta_x, dtype=np.float64))
        centered_y = delta_y - batch_mean_y
        centered_x = delta_x - batch_mean_x
        batch_m2_y = np.sum(
            centered_y * centered_y,
            axis=0,
            dtype=np.float64,
        )
        batch_m2_x = float(np.dot(centered_x, centered_x))
        batch_covariance = np.sum(
            centered_y * centered_x[:, None],
            axis=0,
            dtype=np.float64,
        )

        if count == 0:
            count = current
            mean_delta_y = batch_mean_y
            mean_delta_x = batch_mean_x
            m2_delta_y = batch_m2_y
            m2_delta_x = batch_m2_x
            covariance = batch_covariance
        else:
            total = count + current
            delta_mean_y = batch_mean_y - mean_delta_y
            delta_mean_x = batch_mean_x - mean_delta_x
            cross_weight = count * current / total
            m2_delta_y += (
                batch_m2_y
                + delta_mean_y * delta_mean_y * cross_weight
            )
            m2_delta_x += (
                batch_m2_x
                + delta_mean_x * delta_mean_x * cross_weight
            )
            covariance += (
                batch_covariance
                + delta_mean_y * delta_mean_x * cross_weight
            )
            mean_delta_y += delta_mean_y * current / total
            mean_delta_x += delta_mean_x * current / total
            count = total

        sum_y_plus += np.sum(y_plus, axis=0, dtype=np.float64)
        sum_y_minus += np.sum(y_minus, axis=0, dtype=np.float64)
        sum_x_plus += float(np.sum(x_plus, dtype=np.float64))
        sum_x_minus += float(np.sum(x_minus, dtype=np.float64))

        batch_mean_base_y = float(np.mean(y_base, dtype=np.float64))
        batch_mean_base_x = float(np.mean(x_base, dtype=np.float64))
        centered_base_y = y_base - batch_mean_base_y
        centered_base_x = x_base - batch_mean_base_x
        batch_m2_base_x = float(np.dot(centered_base_x, centered_base_x))
        batch_cov_base = float(np.dot(centered_base_x, centered_base_y))
        if baseline_count == 0:
            baseline_count = current
            baseline_mean_y = batch_mean_base_y
            baseline_mean_x = batch_mean_base_x
            baseline_m2_x = batch_m2_base_x
            baseline_covariance = batch_cov_base
        else:
            baseline_total = baseline_count + current
            base_delta_y = batch_mean_base_y - baseline_mean_y
            base_delta_x = batch_mean_base_x - baseline_mean_x
            base_weight = baseline_count * current / baseline_total
            baseline_m2_x += (
                batch_m2_base_x + base_delta_x * base_delta_x * base_weight
            )
            baseline_covariance += (
                batch_cov_base + base_delta_x * base_delta_y * base_weight
            )
            baseline_mean_y += base_delta_y * current / baseline_total
            baseline_mean_x += base_delta_x * current / baseline_total
            baseline_count = baseline_total
        remaining -= current

    if m2_delta_x <= 0.0:
        raise RuntimeError("Geometric-control Delta has zero sample variance.")
    beta = covariance / m2_delta_x
    mean_y_plus = sum_y_plus / count
    mean_y_minus = sum_y_minus / count
    mean_x_plus = sum_x_plus / count
    mean_x_minus = sum_x_minus / count
    price_plus = mean_y_plus - beta * (mean_x_plus - known_plus)
    price_minus = mean_y_minus - beta * (mean_x_minus - known_minus)
    delta = (price_plus - price_minus) / (2.0 * bump)

    adjusted_m2 = (
        m2_delta_y
        + beta * beta * m2_delta_x
        - 2.0 * beta * covariance
    )
    adjusted_variance = np.maximum(
        0.0,
        adjusted_m2 / (count - 1),
    )
    raw_variance = m2_delta_y / (count - 1)
    standard_error = np.sqrt(adjusted_variance / count)
    raw_standard_error = np.sqrt(raw_variance / count)
    baseline_beta = baseline_covariance / baseline_m2_x
    known_baseline = geometric_call_price(
        spot,
        strike,
        rate,
        sigma,
        maturity,
        dim,
    )
    baseline_price = baseline_mean_y - baseline_beta * (
        baseline_mean_x - known_baseline
    )

    return DeltaEstimate(
        dim=dim,
        bump=bump,
        paths=paths,
        seed=seed,
        delta=delta,
        standard_error=standard_error,
        raw_standard_error=raw_standard_error,
        beta=beta,
        price_plus=price_plus,
        price_minus=price_minus,
        baseline_price=baseline_price,
        elapsed_seconds=perf_counter() - started,
    )
