from __future__ import print_function
from pls_dev import pls
import numpy as np


np.random.seed(42) #seed for the random state to reproduce the test

nobs = 500
nvars = 100

X = np.random.rand(nobs,nvars)
y = np.random.rand(nobs)

PLS = pls()
result = PLS.varSelectionFFD(X,y,A=2, autoscale = True)

print(result)