import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

true_mu = 3
true_sigma = 1

data = np.random.normal(true_mu, true_sigma, size=100)
#print(data)

# define prior parameters
mu0 = 0
lambda0 = 1

a0 = 1
b0 = 1

#variational family
pass

#free energy
pass

print("Hello, Active Inference!")