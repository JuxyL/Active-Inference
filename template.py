import numpy as np
import matplotlib.pyplot as plt

from scipy.special import digamma, gammaln
from scipy.optimize import minimize
from scipy.stats import norm

np.random.seed(114)

# ---------- True parameters ----------
mu_true = 3.0
sigma_true = 1.0

# precision
# tau = 1 / variance
# variance = sigma^2

tau_true = 1.0 / sigma_true**2

# ---------- Generate data ----------
N = 100

x = np.random.normal(mu_true, sigma_true, N)

# ---------- Prior hyperparameters ----------

mu0 = 0.0
lambda0 = 1.0

a0 = 2.0
b0 = 2.0

def free_energy(params, alpha, x):

        # ---------- unpack ----------
    m, log_s2, log_a, log_b = params
    s2 = np.exp(log_s2)
    a = np.exp(log_a)
    b = np.exp(log_b)

    N = len(x)
    # E[mu]
    E_mu = m

    # E[mu^2]
    E_mu2 = m**2 + s2

    # E[tau]
    E_tau = a / b

    E_logtau = digamma(a) - np.log(b)

    # ==========================================================
    # TERM 1
    # Expected log likelihood
    # ==========================================================

    # sum_i E[(x_i - mu)^2]
    sq_error = np.sum(x**2 - 2*x*E_mu + E_mu2)

    term1 = (
        -0.5 * N * E_logtau
        +0.5 * E_tau * sq_error
    )

    # ==========================================================
    # TERM 2
    # Entropy of q(mu)
    # ==========================================================

    entropy_mu = 0.5 * np.log(2*np.pi*np.e*s2)

    # negative entropy contribution
    term2 = -entropy_mu

    # ==========================================================
    # TERM 3
    # E_q log p(mu | tau)
    # ==========================================================

    # log p(mu|tau)
    # = 1/2 log(lambda0 tau / 2pi)
    #   - lambda0 tau /2 (mu-mu0)^2

    mu_quad = E_mu2 - 2*mu0*E_mu + mu0**2

    term3 = (
        0.5 * (np.log(lambda0) + E_logtau - np.log(2*np.pi))
        -0.5 * lambda0 * E_tau * mu_quad
    )

    # ==========================================================
    # TERM 4
    # Entropy of q(tau)
    # ==========================================================

    entropy_tau = (
        a
        - np.log(b)
        + gammaln(a)
        + (1-a)*digamma(a)
    )

    term4 = -entropy_tau

    # ==========================================================
    # TERM 5
    # E_q log p(tau)
    # ==========================================================

    term5 = (
        a0*np.log(b0)
        - gammaln(a0)
        + (a0-1)*E_logtau
        - b0*E_tau
    )

    # ==========================================================
    # TOTAL FREE ENERGY
    # ==========================================================

    F = term1 + (1/alpha)*(term2 - term3 + term4 - term5)

    return F

#update variational parameters
alphas = [0.1, 1.0, 400.0]

results = {}

for alpha in alphas:

    init = np.array([
        0.0,
        np.log(1.0),
        np.log(2.0),
        np.log(2.0)
    ])

    result = minimize(
        free_energy,
        init,
        args=(alpha, x),
        method='L-BFGS-B'
    )

    m, log_s2, log_a, log_b = result.x

    results[alpha] = {
        'm': m,
        's2': np.exp(log_s2),
        'a': np.exp(log_a),
        'b': np.exp(log_b),
        'F': result.fun
    }

for alpha in alphas:

    r = results[alpha]

    print('\n===========================')
    print(f'alpha = {alpha}')
    print('===========================')

    print(f"posterior mean m      = {r['m']:.4f}")
    print(f"posterior variance s2 = {r['s2']:.4f}")

    print(f"posterior Gamma a     = {r['a']:.4f}")
    print(f"posterior Gamma b     = {r['b']:.4f}")

    print(f"Free Energy           = {r['F']:.4f}")

#visalize the results
mu_grid = np.linspace(-2, 6, 500)

# ---------- prior over mu ----------
prior_std = np.sqrt(1/lambda0)
prior_pdf = norm.pdf(mu_grid, mu0, prior_std)

# ---------- approximate likelihood ----------
# for visualization only

sample_mean = np.mean(x)
sample_std = np.std(x)/np.sqrt(N)

likelihood_pdf = norm.pdf(mu_grid, sample_mean, sample_std)

# ---------- plot ----------

fig, axes = plt.subplots(1, 3, figsize=(18,5))

for ax, alpha in zip(axes, alphas):

    r = results[alpha]

    q_pdf = norm.pdf(mu_grid, r['m'], np.sqrt(r['s2']))

    ax.plot(mu_grid, prior_pdf, label='Prior')
    ax.plot(mu_grid, likelihood_pdf, label='Likelihood')
    ax.plot(mu_grid, q_pdf, label='q posterior')

    ax.set_title(f'alpha = {alpha}')
    ax.legend()

plt.tight_layout()
plt.show()