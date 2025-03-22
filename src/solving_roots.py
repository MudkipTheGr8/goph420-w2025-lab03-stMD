# Name: Matthew Davidson UCID: 30182729
# File Description: Contains functions for solving roots of functions
#
#
#
#
import numpy as np
#
#
# Function Purpose: Finding the root of a function using the secant method
# Parameters:
# - x0: initial guess (float)
# - dx: step size for derivative (float)
# - f: function to find the root of
#
# Returns:
# x_new: estimated root of the function (type: float)
# i+1: number of iterations to convergence (type: int)
# errors: array of errors at each iteration (type: numpy array)
#

def root_secant_modified(x0, dx, f):
    x_prev = x0
    errors = np.zeros(100)
    for i in range(100):
        df = (f(x_prev + dx) - f(x_prev)) / dx
        if abs(df) < 1e-10:
            df = 1e-10 * (1 if df >= 0 else -1)
        x_new = x_prev - f(x_prev) / df
        if abs(x_new) > 1e-10:
            errors[i] = abs((x_new - x_prev) / x_new)
        else:
            errors[i] = abs(x_new - x_prev)
        if errors[i] < 1e-6:
            return x_new, i+1, errors[:i+1]
        x_prev = x_new
    return x_prev, 100, errors