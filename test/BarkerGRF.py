"""
barker_grf_model.py

Standalone implementation of the Barker (1988) Generalized Radial Flow (GRF)
model for constant-rate hydraulic interference tests.

The model supports non-integer flow dimension n and evaluates the Barker GRF
solution by numerical inverse Laplace transformation using the Stehfest method.

References
----------
Barker, J. A. (1988). A generalized radial flow model for hydraulic tests in
fractured rock. Water Resources Research, 24(10), 1796-1804.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gamma, gammaincc, kv, factorial


@dataclass
class BarkerGRF:
    """
    Barker generalized radial flow model.

    Parameters
    ----------
    Q : float, optional
        Pumping rate [m^3/s]. Stored for metadata and later hydraulic parameter
        conversion, but not directly used in the dimensionless drawdown solution.
    r : float
        Radial distance between pumping and observation well [m].
    rw : float
        Well radius [m].
    p : sequence of float, optional
        Model parameters [a, t0, n].

        a  : drawdown scale parameter
        t0 : characteristic time parameter
        n  : flow dimension

    Notes
    -----
    Flow dimension interpretation:

    n = 1      linear flow
    n = 2      classical radial / cylindrical flow
    n = 3      spherical flow
    noninteger n describes generalized or fractal-like flow geometry.
    """

    Q: Optional[float] = None
    r: float = 1.0
    rw: float = 1.0
    p: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.r <= 0.0:
            raise ValueError("r must be positive.")
        if self.rw <= 0.0:
            raise ValueError("rw must be positive.")

        self.r = float(self.r)
        self.rw = float(self.rw)
        self.rD = self.r / self.rw

        if self.p is None:
            self.p = np.array([1.0, 1.0, 2.0], dtype=float)
        else:
            self.p = np.asarray(self.p, dtype=float)

        if self.p.shape != (3,):
            raise ValueError("p must contain exactly three values: [a, t0, n].")

    @property
    def a(self) -> float:
        return float(self.p[0])

    @property
    def t0(self) -> float:
        return float(self.p[1])

    @property
    def n(self) -> float:
        return float(self.p[2])

    def dimensionless_time(self, t: np.ndarray | float) -> np.ndarray:
        """
        Calculate dimensionless time.

        Parameters
        ----------
        t : array-like or float
            Time [s].

        Returns
        -------
        ndarray
            Dimensionless time t_D.
        """
        t = np.asarray(t, dtype=float)

        if self.t0 <= 0.0:
            raise ValueError("t0 must be positive.")

        return t / (2.2458 * self.t0)

    def dimensionless_drawdown_direct(self, td: np.ndarray | float) -> np.ndarray:
        """
        Direct time-domain dimensionless Barker GRF drawdown.

        This expression is useful as a reference, but the Laplace-domain
        solution with numerical inversion is usually more robust across flow
        dimensions.

        Parameters
        ----------
        td : array-like or float
            Dimensionless time.

        Returns
        -------
        ndarray
            Dimensionless drawdown s_D.
        """
        td = np.asarray(td, dtype=float)

        if np.any(td <= 0.0):
            raise ValueError("Dimensionless time td must be positive.")

        alpha = self.n / 2.0 - 1.0
        u = self.rD**2 / (4.0 * td)

        return (
            self.rD ** (2.0 - self.n)
            / (4.0 * np.pi ** (self.n / 2.0))
            * gamma(alpha)
            * gammaincc(alpha, u)
        )

    def dimensionless_drawdown_laplace(self, pd: np.ndarray | float) -> np.ndarray:
        """
        Barker GRF drawdown in Laplace space.

        Parameters
        ----------
        pd : array-like or float
            Laplace-domain parameter p_D.

        Returns
        -------
        ndarray
            Laplace-domain dimensionless drawdown.
        """
        pd = np.asarray(pd, dtype=float)

        if np.any(pd <= 0.0):
            raise ValueError("Laplace parameter pd must be positive.")

        nu = self.n / 2.0 - 1.0

        return (
            self.rD ** (2.0 - self.n)
            * (self.rD**2 * pd / 4.0) ** (self.n / 4.0 - 0.5)
            * kv(nu, self.rD * np.sqrt(pd))
            / pd
            / gamma(self.n / 2.0)
        )

    @staticmethod
    def _stehfest_coefficients(n_terms: int) -> np.ndarray:
        """
        Compute Stehfest coefficients using scipy.special.factorial.

        Parameters
        ----------
        n_terms : int
            Number of Stehfest terms. Must be even.

        Returns
        -------
        ndarray
            Stehfest coefficients V_k for k = 1, ..., n_terms.
        """
        if n_terms <= 0:
            raise ValueError("n_terms must be positive.")
        if n_terms % 2 != 0:
            raise ValueError("n_terms must be even.")

        N = int(n_terms)
        V = np.zeros(N, dtype=float)

        for k in range(1, N + 1):
            j = np.arange((k + 1) // 2, min(k, N // 2) + 1, dtype=float)

            numerator = j ** (N / 2.0) * factorial(2.0 * j, exact=False)

            denominator = (
                factorial(N / 2.0 - j, exact=False)
                * factorial(j, exact=False)
                * factorial(j - 1.0, exact=False)
                * factorial(k - j, exact=False)
                * factorial(2.0 * j - k, exact=False)
            )

            V[k - 1] = (-1.0) ** (k + N / 2.0) * np.sum(numerator / denominator)

        return V

    def inverse_laplace_stehfest(
        self,
        td: np.ndarray | float,
        n_terms: int = 12,
    ) -> np.ndarray:
        """
        Numerically invert the Laplace-domain GRF solution.

        Parameters
        ----------
        td : array-like or float
            Dimensionless time.
        n_terms : int
            Number of Stehfest terms. Must be even. Common values are 8, 10,
            or 12. Larger values are not always more stable.

        Returns
        -------
        ndarray
            Dimensionless drawdown s_D(t_D).
        """
        td = np.asarray(td, dtype=float)
        scalar_input = td.ndim == 0
        td = np.atleast_1d(td)

        V = self._stehfest_coefficients(n_terms)
        k = np.arange(1, n_terms + 1, dtype=float)
        ln2 = np.log(2.0)

        sd = np.full_like(td, np.nan, dtype=float)

        valid = td > 0.0
        for i, tdi in enumerate(td[valid]):
            pd = k * ln2 / tdi
            Fp = self.dimensionless_drawdown_laplace(pd)
            sd[np.flatnonzero(valid)[i]] = ln2 / tdi * np.sum(V * Fp)

        if scalar_input:
            return np.asarray(sd[0])
        return sd

    def dimensional_drawdown(self, sd: np.ndarray | float) -> np.ndarray:
        """
        Convert dimensionless drawdown to dimensional drawdown.

        The factor 0.868588963806504 follows the scaling used in the original
        openhytest-style implementation.
        """
        return np.asarray(sd, dtype=float) * self.a * 0.868588963806504

    def __call__(
        self,
        t: np.ndarray | float,
        n_terms: int = 12,
    ) -> np.ndarray:
        """
        Evaluate dimensional drawdown at time t.

        Parameters
        ----------
        t : array-like or float
            Time [s].
        n_terms : int
            Number of Stehfest terms for inverse Laplace transformation.

        Returns
        -------
        ndarray
            Dimensional drawdown [m].
        """
        td = self.dimensionless_time(t)
        sd = self.inverse_laplace_stehfest(td, n_terms=n_terms)
        return self.dimensional_drawdown(sd)

    def fit(
        self,
        t: Sequence[float],
        s: Sequence[float],
        p0: Optional[Sequence[float]] = None,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        n_terms: int = 12,
        maxfev: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit GRF parameters [a, t0, n] to measured drawdown.

        Parameters
        ----------
        t : sequence of float
            Time data [s].
        s : sequence of float
            Drawdown data [m].
        p0 : sequence of float, optional
            Initial parameter guess [a, t0, n].
        bounds : tuple, optional
            Bounds for scipy.optimize.curve_fit.
        n_terms : int
            Number of Stehfest terms used during fitting.
        maxfev : int
            Maximum number of function evaluations.

        Returns
        -------
        popt : ndarray
            Optimized parameters [a, t0, n].
        pcov : ndarray
            Parameter covariance matrix.
        """
        t = np.asarray(t, dtype=float)
        s = np.asarray(s, dtype=float)

        mask = np.isfinite(t) & np.isfinite(s) & (t > 0.0)
        t_fit = t[mask]
        s_fit = s[mask]

        if t_fit.size < 4:
            raise ValueError("At least four valid data points are required for fitting.")

        if p0 is None:
            p0 = self.p.copy()

        if bounds is None:
            bounds = ([1e-12, 1e-12, 0.5], [np.inf, np.inf, 4.0])

        def model(t_eval: np.ndarray, a: float, t0: float, n: float) -> np.ndarray:
            self.p = np.array([a, t0, n], dtype=float)
            return self(t_eval, n_terms=n_terms)

        popt, pcov = curve_fit(
            model,
            t_fit,
            s_fit,
            p0=np.asarray(p0, dtype=float),
            bounds=bounds,
            maxfev=maxfev,
        )

        self.p = np.asarray(popt, dtype=float)
        return self.p, pcov

    def residuals(
        self,
        t: Sequence[float],
        s: Sequence[float],
        n_terms: int = 12,
    ) -> np.ndarray:
        """Return measured minus calculated drawdown residuals."""
        t = np.asarray(t, dtype=float)
        s = np.asarray(s, dtype=float)
        return s - self(t, n_terms=n_terms)

    def plot(
        self,
        t: Sequence[float],
        s: Optional[Sequence[float]] = None,
        n_terms: int = 12,
        title: str = "Barker GRF model",
    ) -> None:
        """Plot calculated and optionally measured drawdown."""
        t = np.asarray(t, dtype=float)
        s_calc = self(t, n_terms=n_terms)

        fig, ax = plt.subplots(figsize=(8, 6))

        if s is not None:
            ax.loglog(t, s, "o", label="Measured")

        ax.loglog(t, s_calc, "-", label=f"GRF, n = {self.n:.3f}")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Drawdown [m]")
        ax.set_title(title)
        ax.grid(True, which="both")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def plot_type_curves(
        self,
        n_values: Optional[Sequence[float]] = None,
        td_min: float = 1e-1,
        td_max: float = 1e6,
        n_points: int = 120,
        n_terms: int = 12,
    ) -> None:
        """Plot GRF dimensionless type curves for selected flow dimensions."""
        if n_values is None:
            n_values = np.linspace(1.0, 3.0, 9)

        td = np.logspace(np.log10(td_min), np.log10(td_max), n_points) * self.rD**2

        original_p = self.p.copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        for n in n_values:
            self.p = np.array([original_p[0], original_p[1], n], dtype=float)
            sd = self.inverse_laplace_stehfest(td, n_terms=n_terms)
            ax.loglog(td, sd, "-", label=f"n={n:g}")

        self.p = original_p

        ax.set_xlabel("Dimensionless time $t_D$")
        ax.set_ylabel("Dimensionless drawdown $s_D$")
        ax.set_title("Barker GRF type curves")
        ax.grid(True, which="both")
        ax.legend()
        fig.tight_layout()
        plt.show()


def generate_synthetic_data(
    model: BarkerGRF,
    t: np.ndarray,
    noise_std: float = 0.01,
    seed: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate artificial drawdown data from a Barker GRF model.

    Parameters
    ----------
    model : BarkerGRF
        Configured GRF model used to generate synthetic data.
    t : ndarray
        Time vector [s].
    noise_std : float
        Standard deviation of additive Gaussian noise [m].
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    s_true : ndarray
        Noise-free drawdown.
    s_obs : ndarray
        Noisy synthetic observations.
    """
    rng = np.random.default_rng(seed)

    s_true = model(t)
    s_obs = s_true + rng.normal(0.0, noise_std, size=s_true.shape)

    return s_true, s_obs


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Artificial Barker GRF test data and model fit
    # -------------------------------------------------------------------------

    # Time vector [s]
    t_data = np.logspace(0, 5, 100)

    # True model used to generate artificial observations
    true_model = BarkerGRF(
        Q=1.0e-4,
        r=10.0,
        rw=0.05,
        p=[0.5, 1.0, 2.2],
    )

    # Generate artificial data.
    # Use a small noise value so the fitted model is visibly close to the data.
    s_true, s_obs = generate_synthetic_data(
        model=true_model,
        t=t_data,
        noise_std=0.002,
        seed=42,
    )

    # Fit model from an intentionally imperfect initial guess
    fit_model = BarkerGRF(
        Q=1.0e-4,
        r=10.0,
        rw=0.05,
        p=[0.35, 0.6, 2.0],
    )

    popt, pcov = fit_model.fit(
        t=t_data,
        s=s_obs,
        p0=[0.35, 0.6, 2.0],
        bounds=([1e-8, 1e-8, 0.8], [10.0, 1.0e6, 3.5]),
        n_terms=12,
    )

    s_fit = fit_model(t_data, n_terms=12)
    res = s_obs - s_fit
    rmse = np.sqrt(np.mean(res**2))

    print("True parameters")
    print(f"a  = {true_model.a:.6g}")
    print(f"t0 = {true_model.t0:.6g} s")
    print(f"n  = {true_model.n:.6g}")
    print()

    print("Fitted parameters")
    print(f"a  = {popt[0]:.6g}")
    print(f"t0 = {popt[1]:.6g} s")
    print(f"n  = {popt[2]:.6g}")
    print(f"RMSE = {rmse:.6g} m")

    # Plot measured data, true model, and fitted model together
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(t_data, s_obs, "o", markersize=4, label="Artificial observations")
    ax.loglog(t_data, s_true, "--", linewidth=2, label="True GRF model")
    ax.loglog(t_data, s_fit, "-", linewidth=2, label=f"Fitted GRF model, n={fit_model.n:.3f}")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Drawdown [m]")
    ax.set_title("Barker GRF artificial test data and fitted model")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    plt.show()

    # Optional residual plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(t_data, res, "o-", markersize=4)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual s_obs - s_fit [m]")
    ax.set_title("Fit residuals")
    ax.grid(True, which="both")
    fig.tight_layout()
    plt.show()
