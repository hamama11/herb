import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import f2d, grad2d, hess2d

st.title("최적화 알고리즘 비교: 경사하강법 vs 뉴턴 (3D 애니메이션)")
st.markdown("""
### 2D 함수 최적화
함수:  
$$f(x,y) = x^4 - 3x^3 + y^4 - 3y^3 + 2$$  

**최적화 방법 수식:**  
- <span style='color:red'>**경사하강법 (Gradient Descent)**</span>  
$$
x_{t+1} = x_t - \eta \frac{\partial f}{\partial x}, \quad
y_{t+1} = y_t - \eta \frac{\partial f}{\partial y}
$$
- <span style='color:blue'>**뉴턴 방법 (Newton's Method)**</span>  
$$
x_{t+1} = x_t - \frac{\partial f / \partial x}{\partial^2 f / \partial x^2}, \quad
y_{t+1} = y_t - \frac{\partial f / \partial y}{\partial^2 f / \partial y^2}
$$
""", unsafe_allow_html=True)

# -----------------
# 사용자 입력
# -----------------
iterations = st.slider("반복 횟수", 10, 200, 50)
learning_rate = st.slider("학습률 (경사하강법)", 0.01, 1.0, 0.1)
x0 = st.number_input("시작점 x0", value=3.0)
y0 = st.number_input("시작점 y0", value=3.0)

show_gd = st.checkbox("경사하강법", True)
show_newton = st.checkbox("뉴턴 방법", True)

# -----------------
# 최적화 계산
# -----------------
def run_optimization(x0, y0, method):
    x, y = x0, y0
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

history_gd = run_optimization(x0, y0, "Gradient Descent") if show_gd else []
history_newton = run_optimization(x0, y0, "Newton") if show_newton else []

# -----------------
# 3D 표면 + 애니메이션 느낌
# -----------------
X = np.linspace(-1, 3, 100)
Y = np.linspace(-1, 3, 100)
X_grid, Y_grid = np.meshgrid(X, Y)
Z = f2d(X_grid, Y_grid)

fig = go.Figure()

# 표면
fig.add_trace(go.Surface(
    z=Z, x=X_grid, y=Y_grid, colorscale='Viridis', opacity=0.7, showscale=False
))

# 경사하강법 경로
if show_gd and history_gd:
    hx, hy = zip(*history_gd)
    hz = f2d(np.array(hx), np.array(hy))
    fig.add_trace(go.Scatter3d(
        x=hx, y=hy, z=hz,
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=4, color='red'),
        name='경사하강법'
    ))

# 뉴턴 경로
if show_newton and history_newton:
    hx, hy = zip(*history_newton)
    hz = f2d(np.array(hx), np.array(hy))
    fig.add_trace(go.Scatter3d(
        x=hx, y=hy, z=hz,
        mode='lines+markers',
        line=dict(color='blue', width=5),
        marker=dict(size=4, color='blue'),
        name='뉴턴 방법'
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='f(X,Y)'
    ),
    width=900, height=700,
)

st.plotly_chart(fig, use_container_width=True)
