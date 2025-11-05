import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import f1d, grad1d, hess1d, f2d, grad2d, hess2d

st.title("최적화 알고리즘 비교: 경사하강법 vs 뉴턴")
st.markdown("""
- 1D, 2D 함수에서 경사하강법과 뉴턴 방법 비교
- 시작점, 학습률, 반복 횟수 조정 가능
- 수식 표시로 학습 용이
""")

# -----------------
# 사용자 입력
# -----------------
iterations = st.slider("반복 횟수", 10, 500, 50)
learning_rate = st.slider("학습률 (경사하강법)", 0.01, 1.0, 0.1)
x0_1d = st.number_input("1D 시작점 x0", value=3.0)
x0_2d = st.number_input("2D 시작점 x0", value=3.0)
y0_2d = st.number_input("2D 시작점 y0", value=3.0)

# -----------------
# 1D 최적화
# -----------------
st.subheader("1D 함수 최적화")
st.latex(r"f(x) = x^4 - 3x^3 + 2")

def run_1d(x0, method):
    x = x0
    history = [x]
    for _ in range(iterations):
        grad = grad1d(x)
        if method == "경사하강법":
            x = x - learning_rate * grad
        else:
            hess = hess1d(x)
            if hess != 0:
                x = x - grad / hess
        history.append(x)
    return history

history_gd = run_1d(x0_1d, "경사하강법")
history_newton = run_1d(x0_1d, "뉴턴")

X = np.linspace(-1, 3, 400)
Y = f1d(X)

plt.figure(figsize=(8,4))
plt.plot(X, Y, label="f(x)")
plt.plot(history_gd, f1d(np.array(history_gd)), 'ro-', label="경사하강법")
plt.plot(history_newton, f1d(np.array(history_newton)), 'bo-', label="뉴턴")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.title("1D 최적화 비교")
plt.legend()
st.pyplot(plt)

# -----------------
# 2D 최적화
# -----------------
st.subheader("2D 함수 최적화")
st.latex(r"f(x,y) = x^4 - 3x^3 + y^4 - 3y^3 + 2")

def run_2d(x0, y0, method):
    x, y = x0, y0
    history = [(x, y)]
    for _ in range(iterations):
        dfdx, dfdy = grad2d(x, y)
        if method == "경사하강법":
            x -= learning_rate * dfdx
            y -= learning_rate * dfdy
        else:
            hess_x, hess_y = hess2d(x, y)
            x = x - dfdx / hess_x if hess_x != 0 else x
            y = y - dfdy / hess_y if hess_y != 0 else y
        history.append((x, y))
    return history

history_gd2d = run_2d(x0_2d, y0_2d, "경사하강법")
history_newton2d = run_2d(x0_2d, y0_2d, "뉴턴")

# 3D 시각화
Xg = np.linspace(-1, 3, 100)
Yg = np.linspace(-1, 3, 100)
X_grid, Y_grid = np.meshgrid(Xg, Yg)
Z = f2d(X_grid, Y_grid)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.6)

hx_gd, hy_gd = zip(*history_gd2d)
hz_gd = f2d(np.array(hx_gd), np.array(hy_gd))
ax.plot(hx_gd, hy_gd, hz_gd, 'r-o', label="경사하강법")

hx_new, hy_new = zip(*history_newton2d)
hz_new = f2d(np.array(hx_new), np.array(hy_new))
ax.plot(hx_new, hy_new, hz_new, 'b-o', label="뉴턴")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("f(X,Y)")
ax.set_title("2D 최적화 비교")
st.pyplot(fig)
