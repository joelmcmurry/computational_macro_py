"""
Program Name: optimal_growth.py
This program generates the value function and decision rules for a nonstochastic growth model
Adapted from Sargent and Stachurski Quantitative Economics Lectures
"""

import time
import numba
import numpy as np
import matplotlib.pyplot as plt
from numpy import log
from numba import jit
from scipy.optimize import fminbound
from scipy import interp

start_time = time.time()

# Parameters

beta = 0.99 # discount factor
delta = 0.025 # depreciation rate
alpha = 0.36 # capital share

# Asset grid

grid_upper = 45
grid_size = 1800
grid = np.linspace(1e-6,grid_upper,grid_size)

def bellman_operator(w):

    # The Bellman Operator function takes as an input the interpolated value function w
    # defined on the grid points and solves max[u(k,k') + beta*w(k')] where k' is chosen
    # from the interpolated grid points

    # Exploit the monotonicity of the policy rule to only search for k' > k

    Aw = lambda x: interp(x,grid,w)

    Tw = np.empty(grid_size)
    for i, k in enumerate(grid):
        def objective(kprime):
            return - log(k**alpha + (1-delta)*k-kprime) - beta * Aw(kprime)
        kprime_star = fminbound(objective, k, k**alpha+(1-delta)*k)
        Tw[i] = - objective(kprime_star)

    return Tw

 # Find a fixed point of the Bellman Operator (it is a contraction so we are guaranteed existence)

@jit
def fixed_point(T,error_tol,max_iter):

    # Start with an initial guess for the value function and iteratively apply the Bellman Operator
    # until convergence

    w = np.zeros(grid_size)

    iterate = 0
    error = error_tol + 1

    while iterate < max_iter and error > error_tol:
        w_next = bellman_operator(w)
        iterate += 1
        error = np.max(np.abs(w_next - w))
        w = w_next
        print(iterate,error)

    return w_next

v_star = fixed_point(bellman_operator,0.0001,1000)

# Obtain policy function

def policy_function(w):

    # Having found the value function, find decision rule: k' given k

    Aw = lambda x: interp(x,grid,w)

    policy = np.empty(grid_size)
    for i, k in enumerate(grid):
        objective = lambda kprime: - log(k**alpha + (1-delta)*k-kprime) - beta * Aw(kprime)
        policy[i] = fminbound(objective, 1e-6, k**alpha+(1-delta)*k)

    return policy

policyfunction = policy_function(v_star)

elapsed_time = time.time() - start_time
print(elapsed_time)

# Plots

# Value Function

fig, ax = plt.subplots()
ax.set_ylim(30, 120)
ax.set_xlim(np.min(grid), np.max(grid))
lb = 'Value Function'
ax.plot(grid, v_star, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)
ax.legend(loc='upper left')

# Policy Function

fig, ax = plt.subplots()
ax.set_ylim(0, 50)
ax.set_xlim(np.min(grid), np.max(grid))
lb = 'Policy Function'
ax.plot(grid, policyfunction, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)
ax.legend(loc='upper left')

plt.show()