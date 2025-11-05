import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import f1d, grad1d, hess1d, f2d, grad2d, hess2d

# -----------------
# 제목
# -----------------
st.title("최적화 알고리즘 비교: 경사하강법 vs 뉴턴")

# -----------------
# 사용자 입력
# -----------------
iterations = st.slider("반복 횟수", 10, 500, 50)
learning_rate = st.slider("학습률 (경사하강법)", 0.01, 1.0, 0.1)
x0_1d = st.number_input("1D 시작점 x0", value=3.0)
x0_2d = st.number_input("2D 시작점 x0", value=3.0)
y0_2d = st.number_input("2D 시작점 y0", value=3.0)

show_gd = st.checkbox("경사하강법 표시", True)
show_newton = st.checkbox("뉴턴 방법 표시", True)

# -----------------
# 1D 최적화 함수
# -----------------
def run_1d(x0, method):
    x = float(x0)
    history = [x]
    for _ in range(iterations):
        grad = grad1d(x)
        if method == "Gradient Descent":
            x -= learning_rate * grad
        elif method == "Newton":
            hess = hess1d(x)
            x = x - grad / hess if hess != 0 else x
        history.append(x)
    return history

history_gd_1d = run_1d(x0_1d, "Gradient Descent") if show_gd else []
history_newton_1d = run_1d(x0_1d, "Newton") if show_newton else []

# 1D 그래프
X = np.linspace(-1, 3, 400)
Y = f1d(X)
fig1d = go.Figure()
fig1d.add_trace(go.Scatter(x=X, y=Y, mode='lines', name='f(x)'))

if show_gd and history_gd_1d:
    fig1d.add_trace(go.Scatter(
        x=history_gd_1d,
        y=f1d(np.array(history_gd_1d)),
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=6),
        name='경사하강법'
    ))

if show_newton and history_newton_1d:
    fig1d.add_trace(go.Scatter(
        x=history_newton_1d,
        y=f1d(np.array(history_newton_1d)),
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=6),
        name='뉴턴 방법'
    ))

fig1d.update_layout(
    xaxis_title='x',
    yaxis_title='f(x)',
    width=700, height=400,
)

# -----------------
# 2D 최적화 함수
# -----------------
def run_2d(x0, y0, method):
    x, y = float(x0), float(y0)
    history = [(x, y)]
    for _ in range(iterations):
        dfdx, dfdy = grad2d(x, y)
        if method == "Gradient Descent":
            x -= learning_rate * dfdx
            y -= learning_rate * dfdy
        elif method == "Newton":
            hess_x, hess_y = hess2d(x, y)
            x = x - dfdx / hess_x if hess_x != 0 else x
            y = y - dfdy / hess_y if hess_y != 0 else y
        history.append((x, y))
    return history

history_gd_2d = run_2d(x0_2d, y0_2d, "Gradient Descent") if show_gd else []
history_newton_2d = run_2d(x0_2d, y0_2d, "Newton") if show_newton else []

# 2D surface
Xg = np.linspace(-1, 3, 100)
Yg = np.linspace(-1, 3, 100)
X_grid, Y_grid = np.meshgrid(Xg, Yg)
Z = f2d(X_grid, Y_grid)

fig2d = go.Figure()
fig2d.add_trace(go.Surface(x=X_grid, y=Y_grid, z=Z, colorscale='Viridis', opacity=0.7, showscale=False))

if show_gd and history_gd_2d:
    hx, hy = zip(*history_gd_2d)
    hz = f2d(np.array(hx), np.array(hy))
    fig2d.add_trace(go.Scatter3d(
        x=hx, y=hy, z=hz,
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=4),
        name='경사하강법'
    ))

if show_newton and history_newton_2d:
    hx, hy = zip(*history_newton_2d)
    hz = f2d(np.array(hx), np.array(hy))
    fig2d.add_trace(go.Scatter3d(
        x=hx, y=hy, z=hz,
        mode='lines+markers',
        line=dict(color='blue', width=5),
        marker=dict(size=4),
        name='뉴턴 방법'
    ))

fig2d.update_layout(
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='f(X,Y)'
    ),
    width=700, height=400,
)

# -----------------
# 좌우 컬럼 배치
# -----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1D 함수 최적화")
    st.plotly_chart(fig1d, use_container_width=True)

with col2:
    st.subheader("2D 함수 최적화")
    st.plotly_chart(fig2d, use_container_width=True)
