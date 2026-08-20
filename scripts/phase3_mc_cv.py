"""Reusable Monte Carlo engine for Phase 3 arithmetic/geometric baskets."""

from dataclasses import asdict, dataclass
from math import erf, exp, log, sqrt
from time import perf_counter
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class BasketParameters:
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    sigma: float = 0.2
    maturity: float = 1.0
    dim: int = 100


@dataclass
class MCEstimate:
    rho: float
    paths: int
    seed: int
    target: str
    control: str
    analytic_control_price: float
    raw_price: float
    raw_standard_error: float
    raw_ci95_low: float
    raw_ci95_high: float
    raw_ci95_half_width: float
    control_variance: float
    beta: float
    cv_price: float
    cv_standard_error: float
    ci95_low: float
    ci95_high: float
    ci95_half_width: float
    relative_error_pct: float
    relative_ci_half_width_pct: float
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class _BivariateMoments:
    """Numerically stable streaming moments for target Y and control X."""

    def __init__(self) -> None:
        self.count = 0
        self.mean_y = 0.0
        self.mean_x = 0.0
        self.m2_y = 0.0
        self.m2_x = 0.0
        self.c_xy = 0.0

    def update(self, y: np.ndarray, x: np.ndarray) -> None:
        if y.shape != x.shape or y.ndim != 1:
            raise ValueError("Target and control samples must be matching 1D arrays.")
        batch_count = y.size
        if batch_count == 0:
            return

        batch_mean_y = float(np.mean(y, dtype=np.float64))
        batch_mean_x = float(np.mean(x, dtype=np.float64))
        centered_y = y - batch_mean_y
        centered_x = x - batch_mean_x
        batch_m2_y = float(np.dot(centered_y, centered_y))
        batch_m2_x = float(np.dot(centered_x, centered_x))
        batch_c_xy = float(np.dot(centered_x, centered_y))

        if self.count == 0:
            self.count = batch_count
            self.mean_y = batch_mean_y
            self.mean_x = batch_mean_x
            self.m2_y = batch_m2_y
            self.m2_x = batch_m2_x
            self.c_xy = batch_c_xy
            return

        total = self.count + batch_count
        delta_y = batch_mean_y - self.mean_y
        delta_x = batch_mean_x - self.mean_x
        cross_weight = self.count * batch_count / total
        self.m2_y += batch_m2_y + delta_y * delta_y * cross_weight
        self.m2_x += batch_m2_x + delta_x * delta_x * cross_weight
        self.c_xy += batch_c_xy + delta_x * delta_y * cross_weight
        self.mean_y += delta_y * batch_count / total
        self.mean_x += delta_x * batch_count / total
        self.count = total


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def geometric_basket_call_price(params: BasketParameters, rho: float) -> float:
    """Closed-form discounted call price on the geometric average."""
    validate_rho(params.dim, rho)
    variance = (
        params.sigma ** 2
        * params.maturity
        * (1.0 + (params.dim - 1) * rho)
        / params.dim
    )
    std = sqrt(variance)
    log_mean = (
        log(params.spot)
        + (params.rate - 0.5 * params.sigma ** 2) * params.maturity
    )
    d2 = (log_mean - log(params.strike)) / std
    d1 = d2 + std
    discounted_forward_moment = exp(
        -params.rate * params.maturity + log_mean + 0.5 * variance
    )
    discounted_strike = params.strike * exp(-params.rate * params.maturity)
    return (
        discounted_forward_moment * normal_cdf(d1)
        - discounted_strike * normal_cdf(d2)
    )


def validate_rho(dim: int, rho: float) -> None:
    lower_bound = -1.0 / (dim - 1)
    if not lower_bound < rho < 1.0:
        raise ValueError(
            f"Equicorrelation rho must be in ({lower_bound}, 1), got {rho}."
        )


def _discounted_payoffs(
    rng: np.random.Generator,
    params: BasketParameters,
    rho: float,
    num_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return discounted arithmetic and geometric call payoffs."""
    idiosyncratic = rng.standard_normal(
        size=(num_paths, params.dim), dtype=np.float64
    )
    if rho == 0.0:
        shocks = idiosyncratic
    else:
        common = rng.standard_normal(size=(num_paths, 1), dtype=np.float64)
        shocks = sqrt(rho) * common + sqrt(1.0 - rho) * idiosyncratic

    log_terminal = (
        log(params.spot)
        + (params.rate - 0.5 * params.sigma ** 2) * params.maturity
        + params.sigma * sqrt(params.maturity) * shocks
    )
    terminal = np.exp(log_terminal)
    arithmetic_average = np.mean(terminal, axis=1, dtype=np.float64)
    geometric_average = np.exp(np.mean(log_terminal, axis=1, dtype=np.float64))
    discount = exp(-params.rate * params.maturity)
    arithmetic_payoff = discount * np.maximum(
        arithmetic_average - params.strike, 0.0
    )
    geometric_payoff = discount * np.maximum(
        geometric_average - params.strike, 0.0
    )
    return arithmetic_payoff, geometric_payoff


def monte_carlo_with_geometric_control(
    params: BasketParameters,
    rho: float,
    paths: int,
    seed: int,
    target: str,
    chunk_size: int = 25_000,
) -> MCEstimate:
    """Price an arithmetic or geometric basket using the geometric payoff CV."""
    validate_rho(params.dim, rho)
    if target not in {"arithmetic", "geometric"}:
        raise ValueError("target must be 'arithmetic' or 'geometric'.")
    if paths < 2:
        raise ValueError("At least two paths are required.")

    started = perf_counter()
    rng = np.random.default_rng(seed)
    moments = _BivariateMoments()
    remaining = paths
    while remaining:
        current = min(chunk_size, remaining)
        arithmetic, geometric = _discounted_payoffs(
            rng, params, rho, current
        )
        target_payoff = geometric if target == "geometric" else arithmetic
        moments.update(target_payoff, geometric)
        remaining -= current

    if moments.m2_x <= 0.0:
        raise RuntimeError("Geometric control payoff has zero sample variance.")

    beta = moments.c_xy / moments.m2_x
    analytic_control = geometric_basket_call_price(params, rho)
    cv_price = moments.mean_y - beta * (moments.mean_x - analytic_control)

    raw_variance = moments.m2_y / (moments.count - 1)
    control_variance = moments.m2_x / (moments.count - 1)
    adjusted_m2 = (
        moments.m2_y
        + beta * beta * moments.m2_x
        - 2.0 * beta * moments.c_xy
    )
    # Roundoff can make the theoretically non-negative residual a tiny negative.
    adjusted_variance = max(0.0, adjusted_m2 / (moments.count - 1))
    raw_standard_error = sqrt(raw_variance / moments.count)
    raw_ci_half_width = 1.96 * raw_standard_error
    cv_standard_error = sqrt(adjusted_variance / moments.count)
    ci_half_width = 1.96 * cv_standard_error
    reference = (
        analytic_control
        if target == "geometric"
        else cv_price
    )
    relative_error = (
        abs(cv_price - analytic_control) / analytic_control * 100.0
        if target == "geometric"
        else float("nan")
    )

    return MCEstimate(
        rho=rho,
        paths=paths,
        seed=seed,
        target=target,
        control="geometric",
        analytic_control_price=analytic_control,
        raw_price=moments.mean_y,
        raw_standard_error=raw_standard_error,
        raw_ci95_low=moments.mean_y - raw_ci_half_width,
        raw_ci95_high=moments.mean_y + raw_ci_half_width,
        raw_ci95_half_width=raw_ci_half_width,
        control_variance=control_variance,
        beta=beta,
        cv_price=cv_price,
        cv_standard_error=cv_standard_error,
        ci95_low=cv_price - ci_half_width,
        ci95_high=cv_price + ci_half_width,
        ci95_half_width=ci_half_width,
        relative_error_pct=relative_error,
        relative_ci_half_width_pct=ci_half_width / reference * 100.0,
        elapsed_seconds=perf_counter() - started,
    )
