import numpy as np
import tensorflow as tf
from math import erf, exp, log, sqrt


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
        self.np_dtype = np.dtype(getattr(eqn_config, "sample_dtype", "float64"))

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True) / 2

    def g_tf(self, t, x):
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(AllenCahn, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return y - tf.pow(y, 3)

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(x), 1, keepdims=True))


class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """
    def __init__(self, eqn_config):
        super(PricingDefaultRisk, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02   # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
                self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        piecewise_linear = tf.nn.relu(
            tf.nn.relu(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_tf(self, t, x):
        return tf.reduce_min(x, 1, keepdims=True)


class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.mu_bar-(self.sigma**2)/2)*self.delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        temp = tf.reduce_sum(z, 1, keepdims=True) / self.sigma
        return -self.rl * y - (self.mu_bar - self.rl) * temp + (
            (self.rb - self.rl) * tf.maximum(temp - y, 0))

    def g_tf(self, t, x):
        temp = tf.reduce_max(x, 1, keepdims=True)
        return tf.maximum(temp - 120, 0) - 2 * tf.maximum(temp - 150, 0)


class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(BurgersType, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 - 1.0 / (1 + np.exp(0 + np.sum(self.x_init) / self.dim))
        self.sigma = self.dim + 0.0

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return (y - (2 + self.dim) / 2.0 / self.dim) * tf.reduce_sum(z, 1, keepdims=True)

    def g_tf(self, t, x):
        return 1 - 1.0 / (1 + tf.exp(t + tf.reduce_sum(x, 1, keepdims=True) / self.dim))


class QuadraticGradient(Equation):
    """
    An example PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(QuadraticGradient, self).__init__(eqn_config)
        self.alpha = 0.4
        self.x_init = np.zeros(self.dim)
        base = self.total_time + np.sum(np.square(self.x_init) / self.dim)
        self.y_init = np.sin(np.power(base, self.alpha))

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        x_square = tf.reduce_sum(tf.square(x), 1, keepdims=True)
        base = self.total_time - t + x_square / self.dim
        base_alpha = tf.pow(base, self.alpha)
        derivative = self.alpha * tf.pow(base, self.alpha - 1) * tf.cos(base_alpha)
        term1 = tf.reduce_sum(tf.square(z), 1, keepdims=True)
        term2 = -4.0 * (derivative ** 2) * x_square / (self.dim ** 2)
        term3 = derivative
        term4 = -0.5 * (
            2.0 * derivative + 4.0 / (self.dim ** 2) * x_square * self.alpha * (
                (self.alpha - 1) * tf.pow(base, self.alpha - 2) * tf.cos(base_alpha) - (
                    self.alpha * tf.pow(base, 2 * self.alpha - 2) * tf.sin(base_alpha)
                    )
                )
            )
        return term1 + term2 + term3 + term4

    def g_tf(self, t, x):
        return tf.sin(
            tf.pow(tf.reduce_sum(tf.square(x), 1, keepdims=True) / self.dim, self.alpha))


class ReactionDiffusion(Equation):
    """
    Time-dependent reaction-diffusion-type example PDE in Section 4.7 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(ReactionDiffusion, self).__init__(eqn_config)
        self._kappa = 0.6
        self.lambd = 1 / np.sqrt(self.dim)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 + self._kappa + np.sin(self.lambd * np.sum(self.x_init)) * np.exp(
            -self.lambd * self.lambd * self.dim * self.total_time / 2)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        exp_term = tf.exp((self.lambd ** 2) * self.dim * (t - self.total_time) / 2)
        sin_term = tf.sin(self.lambd * tf.reduce_sum(x, 1, keepdims=True))
        temp = y - self._kappa - 1 - sin_term * exp_term
        return tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.square(temp))

    def g_tf(self, t, x):
        return 1 + self._kappa + tf.sin(self.lambd * tf.reduce_sum(x, 1, keepdims=True))


class BlackScholes1D(Equation):
    """One-dimensional European call option under the Black-Scholes model."""
    def __init__(self, eqn_config):
        super(BlackScholes1D, self).__init__(eqn_config)
        if self.dim != 1:
            raise ValueError("BlackScholes1D expects dim=1.")

        # X_t is the stock price S_t in the forward SDE.
        self.x_init = (np.ones(self.dim) * eqn_config.x_init).astype(self.np_dtype)
        self.rate = eqn_config.rate
        self.sigma = eqn_config.sigma
        self.strike = eqn_config.strike

        # y_init is the analytic Black-Scholes value Y_0 used only for reporting.
        self.y_init = self.black_scholes_call_price()

    @staticmethod
    def _normal_cdf(x):
        """Standard normal CDF used by the analytic d1/d2 formula."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def black_scholes_call_price(self):
        """Analytic European call price for comparison with the learned Y_0."""
        spot = float(self.x_init[0])
        maturity = self.total_time
        d1 = (log(spot / self.strike) + (self.rate + 0.5 * self.sigma ** 2) * maturity) / (
            self.sigma * sqrt(maturity))
        d2 = d1 - self.sigma * sqrt(maturity)
        return spot * self._normal_cdf(d1) - self.strike * exp(-self.rate * maturity) * self._normal_cdf(d2)

    def sample(self, num_sample):
        """Sample risk-neutral GBM paths dS_t = r S_t dt + sigma S_t dW_t."""
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        ).astype(self.np_dtype)
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1], dtype=self.np_dtype)
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init

        # Exact discretization keeps S_t positive and matches the lognormal BS model.
        drift = (self.rate - 0.5 * self.sigma ** 2) * self.delta_t
        for i in range(self.num_time_interval):
            diffusion = self.sigma * dw_sample[:, :, i]
            x_sample[:, :, i + 1] = x_sample[:, :, i] * np.exp(drift + diffusion)
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        """BSDE generator f=-rY, so dY_t = rY_t dt + Z_t dW_t."""
        return -self.rate * y

    def g_tf(self, t, x):
        """Terminal payoff g(S_T)=max(S_T-K,0) for a European call."""
        return tf.maximum(x[:, 0:1] - self.strike, tf.zeros_like(x[:, 0:1]))


class GeometricBasket100D(Equation):
    """100-dimensional geometric-average basket call under independent GBMs."""
    def __init__(self, eqn_config):
        super(GeometricBasket100D, self).__init__(eqn_config)
        if self.dim != 100:
            raise ValueError("GeometricBasket100D expects dim=100.")

        # X_t is the vector of stock prices S_t^i in the forward SDE.
        self.x_init = (np.ones(self.dim) * eqn_config.x_init).astype(self.np_dtype)
        self.rate = eqn_config.rate
        self.sigma = eqn_config.sigma
        self.strike = eqn_config.strike

        # y_init is the analytic geometric-basket call value Y_0 for reporting.
        self.y_init = self.geometric_basket_call_price()

    @staticmethod
    def _normal_cdf(x):
        """Standard normal CDF used by the reduced one-dimensional formula."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def geometric_basket_call_price(self):
        """
        Analytic price for max(G_T-K,0), G_T=(prod_i S_T^i)^(1/d).

        For independent assets with common sigma, log(G_T) is normal with
        variance sigma^2 T / d.  The geometric averaging also changes the
        log-drift, so the first lognormal moment is adjusted explicitly.
        """
        maturity = self.total_time
        geom_spot = exp(np.mean(np.log(self.x_init)))
        variance = self.sigma ** 2 * maturity / self.dim
        std = sqrt(variance)
        log_mean = log(geom_spot) + (self.rate - 0.5 * self.sigma ** 2) * maturity
        d2 = (log_mean - log(self.strike)) / std
        d1 = d2 + std
        discounted_forward_moment = exp(-self.rate * maturity + log_mean + 0.5 * variance)
        discounted_strike = self.strike * exp(-self.rate * maturity)
        return discounted_forward_moment * self._normal_cdf(d1) - discounted_strike * self._normal_cdf(d2)

    def sample(self, num_sample):
        """Sample independent risk-neutral GBM paths for all basket components."""
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        ).astype(self.np_dtype)
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1], dtype=self.np_dtype)
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init

        # Exact GBM discretization: dS_t^i = r S_t^i dt + sigma S_t^i dW_t^i.
        drift = (self.rate - 0.5 * self.sigma ** 2) * self.delta_t
        for i in range(self.num_time_interval):
            diffusion = self.sigma * dw_sample[:, :, i]
            x_sample[:, :, i + 1] = x_sample[:, :, i] * np.exp(drift + diffusion)
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        """BSDE generator f=-rY under risk-neutral pricing."""
        return -self.rate * y

    def g_tf(self, t, x):
        """Terminal payoff g(X_T)=max(geometric_average(S_T)-K,0)."""
        log_geometric_average = tf.reduce_mean(tf.math.log(x), 1, keepdims=True)
        geometric_average = tf.exp(log_geometric_average)
        return tf.maximum(geometric_average - self.strike, tf.zeros_like(geometric_average))


class GeometricBasket100DCorrelated(Equation):
    """100D geometric-average basket call with equicorrelated Brownian shocks."""
    def __init__(self, eqn_config):
        super(GeometricBasket100DCorrelated, self).__init__(eqn_config)
        if self.dim != 100:
            raise ValueError("GeometricBasket100DCorrelated expects dim=100.")

        # X_t is the vector of stock prices S_t^i in the forward SDE.
        self.x_init = (np.ones(self.dim) * eqn_config.x_init).astype(self.np_dtype)
        self.rate = eqn_config.rate
        self.sigma = eqn_config.sigma
        self.strike = eqn_config.strike
        self.rho = eqn_config.rho

        if self.rho <= -1.0 / (self.dim - 1) or self.rho >= 1.0:
            raise ValueError("Equicorrelation rho must be in (-1/(dim-1), 1).")

        corr = np.full((self.dim, self.dim), self.rho)
        np.fill_diagonal(corr, 1.0)
        self.cholesky = np.linalg.cholesky(corr)

        # y_init is the analytic correlated geometric-basket call value Y_0.
        self.y_init = self.geometric_basket_call_price()

    @staticmethod
    def _normal_cdf(x):
        """Standard normal CDF used by the reduced one-dimensional formula."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def geometric_basket_call_price(self):
        """
        Analytic price for max(G_T-K,0) with equicorrelated assets.

        Var(log G_T) = sigma^2 T / n^2 * sum_i sum_j rho_ij
                     = sigma^2 T * (1 + (n - 1) rho) / n.
        """
        maturity = self.total_time
        geom_spot = exp(np.mean(np.log(self.x_init)))
        variance = self.sigma ** 2 * maturity * (1.0 + (self.dim - 1) * self.rho) / self.dim
        std = sqrt(variance)
        log_mean = log(geom_spot) + (self.rate - 0.5 * self.sigma ** 2) * maturity
        d2 = (log_mean - log(self.strike)) / std
        d1 = d2 + std
        discounted_forward_moment = exp(-self.rate * maturity + log_mean + 0.5 * variance)
        discounted_strike = self.strike * exp(-self.rate * maturity)
        return discounted_forward_moment * self._normal_cdf(d1) - discounted_strike * self._normal_cdf(d2)

    def sample(self, num_sample):
        """Sample risk-neutral GBM paths with equicorrelated Brownian increments."""
        independent_dw = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        ).astype(self.np_dtype)
        dw_sample = np.einsum('ij,bjt->bit', self.cholesky.astype(self.np_dtype), independent_dw).astype(self.np_dtype)
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1], dtype=self.np_dtype)
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init

        # Exact GBM discretization: dS_t^i = r S_t^i dt + sigma S_t^i dW_t^i.
        drift = (self.rate - 0.5 * self.sigma ** 2) * self.delta_t
        for i in range(self.num_time_interval):
            diffusion = self.sigma * dw_sample[:, :, i]
            x_sample[:, :, i + 1] = x_sample[:, :, i] * np.exp(drift + diffusion)
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        """BSDE generator f=-rY under risk-neutral pricing."""
        return -self.rate * y

    def g_tf(self, t, x):
        """Terminal payoff g(X_T)=max(geometric_average(S_T)-K,0)."""
        log_geometric_average = tf.reduce_mean(tf.math.log(x), 1, keepdims=True)
        geometric_average = tf.exp(log_geometric_average)
        return tf.maximum(geometric_average - self.strike, tf.zeros_like(geometric_average))


class ArithmeticBasket100D(GeometricBasket100DCorrelated):
    """100D arithmetic-average basket call with equicorrelated GBMs."""

    def __init__(self, eqn_config):
        super(ArithmeticBasket100D, self).__init__(eqn_config)
        # Arithmetic baskets have no closed form here.  This reporting reference
        # is the independently generated Block 1 MC+CV ground truth.
        self.y_init = float(eqn_config.ground_truth)

    def sample(self, num_sample):
        """Sample paths, avoiding an identity matrix multiply when rho is zero."""
        if self.rho != 0.0:
            return super(ArithmeticBasket100D, self).sample(num_sample)

        dw_sample = (
            np.random.normal(
                size=[num_sample, self.dim, self.num_time_interval]
            )
            * self.sqrt_delta_t
        ).astype(self.np_dtype)
        x_sample = np.zeros(
            [num_sample, self.dim, self.num_time_interval + 1],
            dtype=self.np_dtype,
        )
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        drift = (self.rate - 0.5 * self.sigma ** 2) * self.delta_t
        for i in range(self.num_time_interval):
            diffusion = self.sigma * dw_sample[:, :, i]
            x_sample[:, :, i + 1] = (
                x_sample[:, :, i] * np.exp(drift + diffusion)
            )
        return dw_sample, x_sample

    def g_tf(self, t, x):
        """Terminal payoff g(X_T)=max(arithmetic_average(S_T)-K,0)."""
        arithmetic_average = tf.reduce_mean(x, 1, keepdims=True)
        return tf.maximum(
            arithmetic_average - self.strike,
            tf.zeros_like(arithmetic_average),
        )


class ArithmeticBasketIndependent(Equation):
    """Dimension-generic arithmetic-average basket call for rho=0 experiments."""

    def __init__(self, eqn_config):
        super(ArithmeticBasketIndependent, self).__init__(eqn_config)
        self.x_init = (np.ones(self.dim) * eqn_config.x_init).astype(self.np_dtype)
        self.rate = eqn_config.rate
        self.sigma = eqn_config.sigma
        self.strike = eqn_config.strike
        self.y_init = float(eqn_config.ground_truth)

    def sample(self, num_sample):
        """Sample independent risk-neutral GBMs using exact time stepping."""
        dw_sample = (
            np.random.normal(
                size=[num_sample, self.dim, self.num_time_interval]
            )
            * self.sqrt_delta_t
        ).astype(self.np_dtype)
        x_sample = np.zeros(
            [num_sample, self.dim, self.num_time_interval + 1],
            dtype=self.np_dtype,
        )
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        drift = (self.rate - 0.5 * self.sigma ** 2) * self.delta_t
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (
                x_sample[:, :, i]
                * np.exp(drift + self.sigma * dw_sample[:, :, i])
            )
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.rate * y

    def g_tf(self, t, x):
        arithmetic_average = tf.reduce_mean(x, 1, keepdims=True)
        return tf.maximum(
            arithmetic_average - self.strike,
            tf.zeros_like(arithmetic_average),
        )
