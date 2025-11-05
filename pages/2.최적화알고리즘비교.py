import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import f1d, grad1d, f2d, grad2d

st.title("경사하강법 vs 뉴턴 방법 시각화")
st.markdown("""
이 앱은 최적화 개념을 시각적으로 학습할 수 있습니다.
- 1D / 2D 함수 선택
- 시작점, 학습률, 반복 횟수 조정 가능
- 경사하강법과 뉴턴방법 비교
""")

# -----------------
# 사용자 입력
# -----------------
func_type = st.selectbox("함수 선택", ["1D", "2D"])
method = st.selectbox("최적화 방법", ["경사하강법", "뉴턴방법"])
iterations = st.slider("반복 횟수", min_value=10, max_value=500, value=50)
learning_rate = st.slider("학습률 (경사하강법용)", min_value=0.01, max_value=1.0, value=0.1)

# -----------------
# 초기값 설정
# -----------------
if func_type == "1D":
    x0 = st.number_input("시작점 x0", value=3.0)
    x = x0
    history = [x]

    for i in range(iterations):
        grad = grad1d(x)
        if method == "경사하강법":
            x = x - learning_rate * grad
        elif method == "뉴턴방법":
            hess = 12*x**2 - 18*x
            if hess != 0:
                x = x - grad / hess
        history.append(x)

    # 시각화
    X = np.linspace(-1, 3, 400)
    Y = f1d(X)
    plt.figure(figsize=(8,4))
    plt.plot(X, Y, label="f(x)")
    plt.plot(history, f1d(np.array(history)), 'ro-', label="최적화 경로")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"1D 함수 최적화 ({method})")
    plt.legend()
    st.pyplot(plt)

# -----------------
# 2D 함수
# -----------------
elif func_type == "2D":
    x0 = st.number_input("시작점 x0", value=3.0)
    y0 = st.number_input("시작점 y0", value=3.0)
    x, y = x0, y0
    history = [(x, y)]

    for i in range(iterations):
        dfdx, dfdy = grad2d(x, y)
        if method == "경사하강법":
            x = x - learning_rate * dfdx
            y = y - learning_rate * dfdy
        elif method == "뉴턴방법":
            hess_x = 12*x**2 - 18*x
            hess_y = 12*y**2 - 18*y
            x = x - dfdx / hess_x if hess_x != 0 else x
            y = y - dfdy / hess_y if hess_y != 0 else y
        history.append((x, y))

    # 3D 시각화
    X = np.linspace(-1, 3, 100)
    Y = np.linspace(-1, 3, 100)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = f2d(X_grid, Y_grid)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.6)
    hx, hy = zip(*history)
    hz = f2d(np.array(hx), np.array(hy))
    ax.plot(hx, hy, hz, 'r-o', label="최적화 경로")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(X,Y)")
    ax.set_title(f"2D 함수 최적화 ({method})")
    st.pyplot(fig)
