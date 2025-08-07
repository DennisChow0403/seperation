import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
from scipy.optimize import minimize
import statsmodels.api as sm


np.random.seed(0)
n = 100
x = np.linspace(-2, 2, n)
y = (x > 0).astype(int)  # perfectly separated: x>0 ⇒ y=1

X = sm.add_constant(x)  # add intercept

# fit logit
model = sm.Logit(y, X)
try:
    result = model.fit(disp=0)
    print(result.summary())
except Exception as e:
    print("Fitting failed (as expected due to separation):", e)

# Wald CI (normal approx)
conf_int_wald = result.conf_int()
print("Wald CI:\n", conf_int_wald)
