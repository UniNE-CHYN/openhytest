import numpy as np
import pandas as pd

from BarkerGRF import (
    BarkerGRF,
    generate_synthetic_data,
)

# -----------------------------------------------------------------------------
# 1. Create artificial pumping test data
# -----------------------------------------------------------------------------

# Time vector [s]
t = np.logspace(0, 5, 100)

# True GRF model
true_model = BarkerGRF(
    Q=1e-4,          # pumping rate [m3/s]
    r=10.0,          # distance between wells [m]
    rw=0.05,         # well radius [m]
    p=[0.5, 1.0, 2.2],   # [a, t0, n]
)

# Generate artificial data
s_true, s_obs = generate_synthetic_data(
    model=true_model,
    t=t,
    noise_std=0.002,
    seed=82,
)

# -----------------------------------------------------------------------------
# 2. Save artificial data
# -----------------------------------------------------------------------------

df = pd.DataFrame({
    "t": t,
    "s": s_obs,
})

df.to_csv("synthetic_grf_test.csv", index=False)

print(df.head())

# -----------------------------------------------------------------------------
# 3. Fit Barker GRF model
# -----------------------------------------------------------------------------

fit_model = BarkerGRF(
    Q=1e-4,
    r=10.0,
    rw=0.05,
    p=[0.35, 0.6, 2.0],   # initial guess
)

popt, pcov = fit_model.fit(
    t=t,
    s=s_obs,
    p0=[0.35, 0.6, 2.0],
)

# -----------------------------------------------------------------------------
# 4. Results
# -----------------------------------------------------------------------------

print("\nTRUE PARAMETERS")
print("----------------")
print(f"a  = {true_model.a:.4f}")
print(f"t0 = {true_model.t0:.4f}")
print(f"n  = {true_model.n:.4f}")

print("\nFITTED PARAMETERS")
print("------------------")
print(f"a  = {popt[0]:.4f}")
print(f"t0 = {popt[1]:.4f}")
print(f"n  = {popt[2]:.4f}")

# -----------------------------------------------------------------------------
# 5. Calculate fitted drawdown
# -----------------------------------------------------------------------------

s_fit = fit_model(t)

# -----------------------------------------------------------------------------
# 6. Plot
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(
    t,
    s_obs,
    "o",
    markersize=4,
    label="Artificial observations",
)

ax.loglog(
    t,
    s_true,
    "--",
    linewidth=2,
    label="True GRF model",
)

ax.loglog(
    t,
    s_fit,
    "-",
    linewidth=2,
    label=f"Fitted GRF model (n={fit_model.n:.3f})",
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Drawdown [m]")

ax.grid(True, which="both")
ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 7. Residuals
# -----------------------------------------------------------------------------

residuals = s_obs - s_fit

fig, ax = plt.subplots(figsize=(8, 4))

ax.semilogx(
    t,
    residuals,
    "o-",
    markersize=4,
)

ax.axhline(0.0)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Residuals [m]")

ax.grid(True, which="both")

plt.tight_layout()
plt.show()
