import numpy as np

# 1D 함수
def f1d(x):
    return x**4 - 3*x**3 + 2

def grad1d(x):
    return 4*x**3 - 9*x**2

# 2D 함수
def f2d(X, Y):
    return (X**4 - 3*X**3) + (Y**4 - 3*Y**3) + 2

def grad2d(X, Y):
    dfdx = 4*X**3 - 9*X**2
    dfdy = 4*Y**3 - 9*Y**2
    return dfdx, dfdy
