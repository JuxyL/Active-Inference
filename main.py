import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import gammaln
from pytensor.tensor import psi
from scipy.stats import norm

np.random.seed(42)

# different values of alpha
alphas = [0.05, 0.1, 1.0, 400.0]

# ---------- True parameters ----------
mu_true = 3.0
sigma_true = 1.0

a0 = 2.0
b0 = 2.0

#Generate data
N = 100
x = np.random.normal(mu_true, sigma_true, N)

# Store optimization results
results = {}

#tau
# only parts of the parameters are needed to construct the free energy, so we can directly compute them here

def free_energy(params, alpha, x):

    # ---------- unpack ----------
    m, log_s2, log_a, log_b = params
    s2 = pt.exp(log_s2)
    a = pt.exp(log_a)
    b = pt.exp(log_b)
    lambda0 = 1
    E_logtau = psi(a) - pt.log(b)#psi? I need to learn more about distributions and their properties

    # E[mu]
    E_mu = m

    # E[mu^2]
    E_mu2 = m**2 + s2

    # E[tau]
    E_tau = a / b

    E_logtau = psi(a) - pt.log(b)
    # Constructing the Free Energy
    #term 1 likelihood
    sq_error = pt.sum(x**2 - 2*x*E_mu + E_mu2)
    term1 = -0.5*N*E_logtau + 0.5*E_tau*sq_error#2\pi term is ignored as constant

    #term2 entropy of q(\mu)
    term2 = -(0.5*pt.log(2*pt.pi*np.e*s2))

    #term3 entropy of q(\tau)
    term3 = a - pt.log(b) + pt.gammaln(a) + (1-a)*psi(a)

    #term4 expected log prior of \mu
    mu0 = 0.0
    mu_quad = E_mu2 - 2*mu0*E_mu + mu0**2
    term4 = (0.5*(pt.log(lambda0) + E_logtau)-0.5*lambda0*E_tau*mu_quad)

    #term5 expected log prior of \tau
    term5 = a0*pt.log(b0)- pt.gammaln(a0)+ (a0-1)*E_logtau- b0*E_tau

    F = term1 + (1/alpha)*(term2 + term3 - term4 - term5)

    return F

#showing the free energy landscape for different values of alpha
for alpha in alphas:
    with pm.Model() as model:
    #start using log to bypass positivity constraint

        m = pm.Normal("m",mu=0,sigma = 10)#mean

        log_s2 = pm.Normal("log_s2",mu=0,sigma=1)#inverse of variance lambda_N

        lambda0 = 1.0

        log_a = pm.Normal("log_a",mu=0,sigma=1)

        log_b = pm.Normal("log_b",mu=0,sigma=1)

        x_data = pm.Data("x_data",x)

        pm.Potential("free_energy", -free_energy([m, log_s2, log_a, log_b], alpha, x_data))

        #Using pymc to minimize
        result = pm.find_MAP()

        results[alpha] = result

        #print(result)

#-------------------------------------
#visualization of the results

mu_grid = np.linspace(-2, 6, 500)

# ============================================================
# Prior over mu
# ============================================================

prior_std = np.sqrt(1.0)

prior_pdf = norm.pdf(
    mu_grid,
    loc=0.0,
    scale=prior_std
)

# ============================================================
# Approximate likelihood visualization
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
    len(alphas),
    figsize=(18,5)
)

for ax, alpha in zip(axes, alphas):

    result = results[alpha]

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

    ax.set_xlabel("mu")

    ax.set_ylabel("density")

    ax.legend()

plt.tight_layout()

plt.show()

# ============================================================
# Print summary
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