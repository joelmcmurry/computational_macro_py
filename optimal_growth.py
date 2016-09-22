# Program Name: optimal_growth.py
# This program generates the value function and decision rules for a nonstochastic growth model
# Date: 9/22/2016

import numpy as np
from numpy import log
from scipy.optimize import fminbound
from scipy import interp

# Parameters
beta = 0.99 # discount factor
delta = 0.025 # depreciation rate
alpha = 0.36 # capital share