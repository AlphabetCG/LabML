"""
CPE 342 Machine Learning - Assignment 2
Fitting  y_hat = c0 + c1 * exp(c2 * x)  by (batch) gradient descent.

Tasks 1-3 (loss, gradients, update rules) are implemented below as
loss() / gradient() / the update step inside gradient_descent().
Tasks 4-5 produce the fitted coefficients and the loss-vs-iteration plot.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- data
def load_data(path="data.csv"):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def predict(c, x):
    """y_hat = c0 + c1 * exp(c2 * x)"""
    return c[0] + c[1] * np.exp(c[2] * x)


# ------------------------------------------------- Task 1: loss function
def loss(c, x, y):
    """Mean squared error:  J = (1/n) * sum( (y_i - y_hat_i)^2 )"""
    r = y - predict(c, x)
    return np.mean(r ** 2)


# --------------------------------------------------- Task 2: gradients
def gradient(c, x, y):
    """
    r_i = y_i - (c0 + c1*exp(c2*x_i))

    dJ/dc0 = -(2/n) * sum( r_i )
    dJ/dc1 = -(2/n) * sum( r_i * exp(c2*x_i) )
    dJ/dc2 = -(2/n) * sum( r_i * c1 * x_i * exp(c2*x_i) )
    """
    n = len(x)
    e = np.exp(c[2] * x)
    r = y - (c[0] + c[1] * e)
    g0 = -(2.0 / n) * np.sum(r)
    g1 = -(2.0 / n) * np.sum(r * e)
    g2 = -(2.0 / n) * np.sum(r * c[1] * x * e)
    return np.array([g0, g1, g2])


def numerical_gradient(c, x, y, h=1e-6):
    """Finite-difference check that the analytic gradient above is correct."""
    g = np.zeros(3)
    for k in range(3):
        cp, cm = c.astype(float).copy(), c.astype(float).copy()
        cp[k] += h
        cm[k] -= h
        g[k] = (loss(cp, x, y) - loss(cm, x, y)) / (2 * h)
    return g


# ------------------------------------------- Task 3+4: the update rules
def gradient_descent(x, y, c_init, alpha=0.01, n_iter=200_000, tol=1e-12):
    """
    Update rule (applied simultaneously to all three coefficients):

        c0 <- c0 - alpha * dJ/dc0
        c1 <- c1 - alpha * dJ/dc1
        c2 <- c2 - alpha * dJ/dc2
    """
    c = np.array(c_init, dtype=float)
    history = [loss(c, x, y)]
    for i in range(n_iter):
        c = c - alpha * gradient(c, x, y)          # <-- update rule
        history.append(loss(c, x, y))
        if abs(history[-2] - history[-1]) < tol:
            print(f"converged at iteration {i + 1} (dJ < {tol})")
            break
    return c, np.array(history)


# ------------------------------------------------------------- main
if __name__ == "__main__":
    x, y = load_data()
    print(f"n = {len(x)},  x in [{x.min():.3f}, {x.max():.3f}],  "
          f"y in [{y.min():.3f}, {y.max():.3f}]")

    c_init = np.array([1.0, 1.0, -1.0])   # c2 MUST start negative: the data decays
    alpha = 0.01

    # sanity check of Task 2
    ga = gradient(c_init, x, y)
    gn = numerical_gradient(c_init, x, y)
    print(f"analytic  gradient at init: {ga}")
    print(f"numerical gradient at init: {gn}")
    print(f"max abs difference        : {np.max(np.abs(ga - gn)):.3e}\n")

    c, hist = gradient_descent(x, y, c_init, alpha=alpha)

    print(f"\ninitial loss : {hist[0]:.8f}")
    print(f"final   loss : {hist[-1]:.8f}   after {len(hist) - 1} iterations")
    print(f"\nfitted coefficients (alpha = {alpha}):")
    print(f"  c0 = {c[0]:.6f}")
    print(f"  c1 = {c[1]:.6f}")
    print(f"  c2 = {c[2]:.6f}")
    print(f"\n  y_hat = {c[0]:.4f} + {c[1]:.4f} * exp({c[2]:.4f} * x)")

    rmse = np.sqrt(hist[-1])
    print(f"  RMSE = {rmse:.6f}")

    # loss at a few checkpoints, for the report table
    print("\n  iteration      loss")
    for k in [0, 1, 10, 100, 1_000, 10_000, 50_000, 100_000, len(hist) - 1]:
        if k < len(hist):
            print(f"  {k:>9,}  {hist[k]:.8f}")

    # ------------------------------------------------------ Task 5 plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(hist, lw=1.5)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss  J (MSE)")
    ax[0].set_yscale("log")
    ax[0].set_title("Loss over iterations (log scale)")
    ax[0].grid(alpha=0.3)

    xs = np.linspace(x.min(), x.max(), 300)
    ax[1].scatter(x, y, s=12, label="data")
    ax[1].plot(xs, predict(c, xs), "r-", lw=2,
               label=f"$\\hat{{y}} = {c[0]:.3f} + {c[1]:.3f}e^{{{c[2]:.3f}x}}$")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Fitted model")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("result.png", dpi=150)
    print("\nsaved result.png")
