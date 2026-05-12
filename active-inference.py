import numpy as np
import matplotlib.pyplot as plt

import pymc as pm
import pytensor.tensor as pt

from pytensor.tensor import gammaln
from pytensor.tensor import psi

from scipy.stats import norm

# ============================================================
# 1. Generate synthetic Gaussian data
# ============================================================

np.random.seed(42)

# True hidden parameters
mu_true = 3.0
sigma_true = 1.0

# Number of observations
N = 100

# Observed data
x = np.random.normal(mu_true, sigma_true, N)

# ============================================================
# 2. Prior hyperparameters
# ============================================================

# Prior over mu | tau
mu0 = 0.0
lambda0 = 1.0

# Prior over tau
a0 = 2.0
b0 = 2.0

# ============================================================
# 3. Rationality parameters
# ============================================================

alphas = [0.1, 1.0, 400.0]

# Store optimization results
results = {}

# ============================================================
# 4. Loop over alpha values
# ============================================================

for alpha in alphas:

    print("\n====================================")
    print(f"Running alpha = {alpha}")
    print("====================================")

    # --------------------------------------------------------
    # Build a fresh PyMC model
    # --------------------------------------------------------

    with pm.Model() as model:

        # ====================================================
        # Variational parameters
        #
        # q(mu)  = Normal(m, s²)
        # q(tau) = Gamma(a,b)
        #
        # We optimize:
        #   m
        #   log_s2
        #   log_a
        #   log_b
        #
        # to guarantee positivity after exponentiation
        # ====================================================

        m = pm.Normal("m", mu=0.0, sigma=10.0)

        log_s2 = pm.Normal("log_s2", mu=0.0, sigma=2.0)

        log_a = pm.Normal("log_a", mu=0.0, sigma=2.0)

        log_b = pm.Normal("log_b", mu=0.0, sigma=2.0)

        # ----------------------------------------------------
        # Positive transforms
        # ----------------------------------------------------

        s2 = pt.exp(log_s2)

        a = pt.exp(log_a)

        b = pt.exp(log_b)

        # ====================================================
        # Expectations under q
        # ====================================================

        # E_q[mu]
        E_mu = m

        # E_q[mu²]
        E_mu2 = m**2 + s2

        # E_q[tau]
        E_tau = a / b

        # E_q[log tau]
        E_logtau = psi(a) - pt.log(b)

        # ====================================================
        # TERM 1
        #
        # -E_q[ log p(D | mu, tau) ]
        # ====================================================

        x_data = pm.Data("x_data", x)

        # E[(x_i - mu)^2]
        sq_error = pt.sum(
            x_data**2
            - 2 * x_data * E_mu
            + E_mu2
        )

        term1 = (
            -0.5 * N * E_logtau
            + 0.5 * E_tau * sq_error
        )

        # ====================================================
        # TERM 2
        #
        # E_q[ log q(mu) ]
        #
        # Gaussian entropy:
        # H = 1/2 log(2π e s²)
        #
        # Therefore:
        # E[log q] = -H
        # ====================================================

        entropy_mu = (
            0.5 * pt.log(2 * np.pi * np.e * s2)
        )

        term2 = -entropy_mu

        # ====================================================
        # TERM 3
        #
        # E_q[ log q(tau) ]
        #
        # Gamma entropy
        # ====================================================

        entropy_tau = (
            a
            - pt.log(b)
            + gammaln(a)
            + (1 - a) * psi(a)
        )

        term3 = -entropy_tau

        # ====================================================
        # TERM 4
        #
        # E_q[ log p(mu | tau) ]
        #
        # Prior:
        #
        # p(mu|tau)
        # = Normal(mu0, (lambda0 tau)^-1)
        # ====================================================

        mu_quad = (
            E_mu2
            - 2 * mu0 * E_mu
            + mu0**2
        )

        term4 = (
            0.5 * (
                pt.log(lambda0)
                + E_logtau
            )
            - 0.5 * lambda0 * E_tau * mu_quad
        )

        # ====================================================
        # TERM 5
        #
        # E_q[ log p(tau) ]
        #
        # Prior:
        #
        # p(tau) = Gamma(a0,b0)
        # ====================================================

        term5 = (
            a0 * pt.log(b0)
            - gammaln(a0)
            + (a0 - 1) * E_logtau
            - b0 * E_tau
        )

        # ====================================================
        # FREE ENERGY
        #
        # F
        # =
        # accuracy
        # +
        # (1/alpha) complexity
        #
        # complexity = KL(q||p)
        # ====================================================

        F = (
            term1
            + (1 / alpha)
            * (
                term2
                + term3
                - term4
                - term5
            )
        )

        # ====================================================
        # PyMC maximizes logp
        #
        # We want to minimize F
        #
        # Therefore:
        #
        # logp = -F
        # ====================================================

        pm.Potential("free_energy", -F)

        # ====================================================
        # Optimize variational parameters
        # ====================================================

        result = pm.find_MAP()

        results[alpha] = result

        print(result)

# ============================================================
# 5. Plot variational posterior q(mu)
# ============================================================

mu_grid = np.linspace(-2, 6, 500)

# ------------------------------------------------------------
# Prior over mu
# ------------------------------------------------------------

prior_std = np.sqrt(1 / lambda0)

prior_pdf = norm.pdf(
    mu_grid,
    loc=mu0,
    scale=prior_std
)

# ------------------------------------------------------------
# Approximate likelihood visualization
#
# NOTE:
# This is only for plotting intuition
# ============================================================

sample_mean = np.mean(x)

sample_std = np.std(x) / np.sqrt(N)

likelihood_pdf = norm.pdf(
    mu_grid,
    loc=sample_mean,
    scale=sample_std
)

# ============================================================
# Plot posterior q(mu)
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, alpha in zip(axes, alphas):

    result = results[alpha]

    # Recover posterior variance
    s2 = np.exp(result["log_s2"])

    q_pdf = norm.pdf(
        mu_grid,
        loc=result["m"],
        scale=np.sqrt(s2)
    )

    ax.plot(
        mu_grid,
        prior_pdf,
        label="Prior"
    )

    ax.plot(
        mu_grid,
        likelihood_pdf,
        label="Likelihood"
    )

    ax.plot(
        mu_grid,
        q_pdf,
        label="Variational Posterior q"
    )

    ax.set_title(f"alpha = {alpha}")

    ax.legend()

plt.tight_layout()

plt.show()

# ============================================================
# 6. Print summary
# ============================================================

print("\n====================================")
print("FINAL RESULTS")
print("====================================")

for alpha in alphas:

    r = results[alpha]

    print("\n------------------------------------")
    print(f"alpha = {alpha}")
    print("------------------------------------")

    print("m =", r["m"])

    print("s² =", np.exp(r["log_s2"]))

    print("a =", np.exp(r["log_a"]))

    print("b =", np.exp(r["log_b"]))