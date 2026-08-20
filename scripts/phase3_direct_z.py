"""Direct analytic-Z supervision for the preregistered 1D experiment.

This module is intentionally separate from ``solver.py`` so the frozen Phase 3
baseline implementation is not changed.  In particular, ``lambda_z == 0``
delegates to the original loss implementation without introducing additional
forward passes or BatchNorm updates.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from solver import BSDESolver, DELTA_CLIP


def normal_cdf_tf(value: tf.Tensor) -> tf.Tensor:
    return tf.constant(0.5, value.dtype) * (
        tf.constant(1.0, value.dtype)
        + tf.math.erf(value / tf.sqrt(tf.constant(2.0, value.dtype)))
    )


def analytic_delta_tf(
    states: tf.Tensor,
    time_value: float,
    *,
    maturity: float,
    strike: float,
    rate: float,
    sigma: float,
) -> tf.Tensor:
    """Black--Scholes call Delta at a nonterminal left endpoint."""
    states = tf.convert_to_tensor(states)
    tau = tf.constant(maturity - time_value, states.dtype)
    if maturity - time_value <= 0.0:
        raise ValueError("Analytic supervision is undefined at terminal time.")
    sigma_tf = tf.constant(sigma, states.dtype)
    d1 = (
        tf.math.log(states / tf.constant(strike, states.dtype))
        + tf.constant(rate + 0.5 * sigma**2, states.dtype) * tau
    ) / (sigma_tf * tf.sqrt(tau))
    return normal_cdf_tf(d1)


def analytic_z_tf(states: tf.Tensor, time_value: float, problem) -> tf.Tensor:
    delta = analytic_delta_tf(
        states,
        time_value,
        maturity=problem.total_time,
        strike=problem.strike,
        rate=problem.rate,
        sigma=problem.sigma,
    )
    return tf.constant(problem.sigma, states.dtype) * states * delta


def deterministic_time_indices(
    scheme: str,
    num_time_interval: int,
    seed: int,
    global_step: tf.Tensor | int,
) -> tf.Tensor:
    """Return preregistered supervision times, with stateless random-five."""
    if scheme == "all-time":
        return tf.range(num_time_interval, dtype=tf.int32)
    if scheme == "near-t0":
        return tf.range(5, dtype=tf.int32)
    if scheme != "random-five":
        raise ValueError(f"Unknown time-supervision scheme: {scheme}")
    step = tf.cast(global_step, tf.int32)
    stateless_seed = tf.stack([tf.cast(seed, tf.int32), step])
    shuffled = tf.random.experimental.stateless_shuffle(
        tf.range(num_time_interval, dtype=tf.int32), seed=stateless_seed
    )
    return tf.sort(shuffled[:5])


class DirectZSolver(BSDESolver):
    """Original terminal objective plus normalized analytic-Z supervision."""

    def __init__(
        self,
        config,
        bsde,
        *,
        lambda_z: float,
        time_scheme: str = "all-time",
        time_sampling_seed: int = 7101,
    ):
        super().__init__(config, bsde)
        if bsde.dim != 1:
            raise ValueError("The current preregistered direct-Z runner is 1D only.")
        if lambda_z < 0:
            raise ValueError("lambda_z must be nonnegative.")
        deterministic_time_indices(time_scheme, bsde.num_time_interval, time_sampling_seed, 0)
        self.lambda_z = float(lambda_z)
        self.time_scheme = time_scheme
        self.time_sampling_seed = int(time_sampling_seed)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="direct_z_global_step")

    def forward_with_z(self, inputs, training):
        """Replay NonsharedModel.call once while retaining each Z actually used."""
        dw, x = inputs
        dtype = self.net_config.dtype
        times = np.arange(self.bsde.num_time_interval) * self.bsde.delta_t
        ones = tf.ones(tf.stack([tf.shape(dw)[0], 1]), dtype=dtype)
        y = ones * self.model.y_init
        z = tf.matmul(ones, self.model.z_init)
        z_values = [z]
        for time_index in range(self.bsde.num_time_interval - 1):
            y = (
                y
                - self.bsde.delta_t
                * self.bsde.f_tf(times[time_index], x[:, :, time_index], y, z)
                + tf.reduce_sum(z * dw[:, :, time_index], axis=1, keepdims=True)
            )
            z = self.model.subnet[time_index](
                x[:, :, time_index + 1], training=training
            ) / self.bsde.dim
            z_values.append(z)
        y = (
            y
            - self.bsde.delta_t
            * self.bsde.f_tf(times[-1], x[:, :, -2], y, z)
            + tf.reduce_sum(z * dw[:, :, -1], axis=1, keepdims=True)
        )
        return y, z_values

    def loss_components(self, inputs, training):
        dw, states = inputs
        terminal, z_values = self.forward_with_z(inputs, training=training)
        payoff = self.bsde.g_tf(self.bsde.total_time, states[:, :, -1])
        residual = terminal - payoff
        terminal_loss = tf.reduce_mean(
            tf.where(
                tf.abs(residual) < DELTA_CLIP,
                tf.square(residual),
                2 * DELTA_CLIP * tf.abs(residual) - DELTA_CLIP**2,
            )
        )
        indices = deterministic_time_indices(
            self.time_scheme,
            self.bsde.num_time_interval,
            self.time_sampling_seed,
            self.global_step,
        )
        per_time_losses = []
        for time_index in range(self.bsde.num_time_interval):
            state_t = states[:, :, time_index]
            target_z = analytic_z_tf(
                state_t, time_index * self.bsde.delta_t, self.bsde
            )
            numerator = tf.reduce_mean(tf.square(z_values[time_index] - target_z))
            denominator = tf.stop_gradient(tf.reduce_mean(tf.square(target_z))) + tf.cast(1e-8, state_t.dtype)
            per_time_losses.append(numerator / denominator)
        all_time_losses = tf.stack(per_time_losses)
        z_loss = tf.reduce_mean(tf.gather(all_time_losses, indices))
        total_loss = terminal_loss + tf.cast(self.lambda_z, terminal_loss.dtype) * z_loss
        return total_loss, terminal_loss, z_loss, indices, terminal, z_values

    def loss_fn(self, inputs, training):
        # This branch is deliberately Python-static and exactly preserves the
        # old runner's operation/RNG/BatchNorm sequence for the compatibility canary.
        if self.lambda_z == 0.0:
            return super().loss_fn(inputs, training)
        return self.loss_components(inputs, training)[0]

    @tf.function
    def train_step(self, train_data):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(train_data, training=True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.global_step.assign_add(1)


def black_scholes_price_np(states, time_value, *, maturity, strike, rate, sigma):
    states = np.asarray(states, dtype=np.float64)
    tau = maturity - time_value
    d1 = (np.log(states / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    cdf = np.vectorize(lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))
    return states * cdf(d1) - strike * math.exp(-rate * tau) * cdf(d2)
