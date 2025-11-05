##.2 최적화 알고리즘 비교: 경사하강법 vs 뉴턴 사용 함수

import numpy as np

# 1D 함수
def f1d(x):
    return x**4 - 3*x**3 + 2

def grad1d(x):
    return 4*x**3 - 9*x**2

def hess1d(x):
    return 12*x**2 - 18*x

# 2D 함수
def f2d(X, Y):
    return (X**4 - 3*X**3) + (Y**4 - 3*Y**3) + 2

def grad2d(X, Y):
    return 4*X**3 - 9*X**2, 4*Y**3 - 9*Y**2

def hess2d(X, Y):
    return 12*X**2 - 18*X, 12*Y**2 - 18*Y
