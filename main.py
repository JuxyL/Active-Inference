import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import gammaln
from pytensor.tensor import psi

np.random.seed(42)

# different values of alpha
alphas = [0.1, 1.0, 400.0]

# ---------- True parameters ----------
mu_true = 3.0
sigma_true = 1.0

#Generate data
N = 100
x = np.random.normal(mu_true, sigma_true, N)

#setting prior parameters
with pm.Model() as model:
#start using log to bypass positivity constraint

    m = pm.Normal("m",0,10)#mean

    log_s2 = pm.Normal("log_s2",0,1)#inverse of variance lambda_N

    log_lambda0 = -log_s2

    log_a = pm.Normal("log_a",0,1)

    log_b = pm.Normal("log_b",0,1)

#tau
# only parts of the parameters are needed to construct the free energy, so we can directly compute them here

x_data = pm.Data("x_data",x)

def free_energy(params, alpha, x):

    # ---------- unpack ----------
    m, log_s2, log_a, log_b = params
    s2 = pt.exp(log_s2)
    a = pt.exp(log_a)
    b = pt.exp(log_b)
    lambda0 = pt.exp(log_lambda0)
    
    E_logtau = psi(a) - pt.log(b)#psi? I need to learn more about distributions and their properties

    N = len(x)
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
    term1 = 0.5*N*E_logtau -2*np.pi - 0.5*E_tau*sq_error#2\pi term is ignored as constant

    #term2 entropy of q(\mu)
    term2 = 0.5*pt.log(2*np.pi*s2) + 0.5

    #term3 entropy of q(\tau)
    term3 = a - pt.log(b) + pt.gammaln(a) + (1-a)*psi(a)

    #term4 expected log prior of \mu
    term4 = 0.5*N*(pt.log(lambda0) + E_logtau -2*np.pi) - 0.5*lambda0*E_tau*E_logtau# what is \tau? I used expected value of \tau here, but is it correct?

    #term5 expected log prior of \tau
    term5 = a*np.log(b)- pt.gammaln(a)+ (a-1)*E_logtau- b*E_tau

    F = term1 + (1/alpha)*(term2 + term3 + term4 + term5)

    return F

#showing the free energy landscape for different values of alpha
for alpha in alphas:
    pm.Potential("free_energy",-free_energy([m, log_s2, log_a, log_b], 1.0, x_data))

    #Using pymc to minimize
    with model:
        result = pm.find_MAP()
    print(result)