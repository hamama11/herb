import streamlit as st
import numpy as np
from scipy.optimize import minimize, differential_evolution
import plotly.graph_objects as go

st.title("🧮 다변수 최적화")

# 함수 선택
func_option = st.selectbox("목적 함수 선택", ["(x-2)^2 + (y-3)^2", "sin(x)*cos(y) + x + y"])

# 변수 범위
x_min, x_max = st.number_input("x 최소값", 0.0), st.number_input("x 최대값", 5.0)
y_min, y_max = st.number_input("y 최소값", 0.0), st.number_input("y 최대값", 5.0)

# 최적화 방식
method = st.selectbox("최적화 방식", ["Local (minimize)", "Global (differential_evolution)"])

# 목적 함수 정의
def objective(vars):
    x, y = vars
    if func_option == "(x-2)^2 + (y-3)^2":
        return (x-2)**2 + (y-3)**2
    else:
        return np.sin(x)*np.cos(y) + x + y

# 최적화 버튼
if st.button("최적화 수행"):
    if method == "Local (minimize)":
        res = minimize(objective, x0=[(x_min+x_max)/2, (y_min+y_max)/2],
                       bounds=[(x_min,x_max),(y_min,y_max)])
    else:
        res = differential_evolution(objective, bounds=[(x_min,x_max),(y_min,y_max)])
    
    st.success(f"최적값: {res.fun:.4f}, 최적 변수: x={res.x[0]:.4f}, y={res.x[1]:.4f}")

    # 3D 시각화
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda x, y: objective([x, y]))(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.add_trace(go.Scatter3d(
        x=[res.x[0]], y=[res.x[1]], z=[res.fun],
        mode='markers', marker=dict(size=5, color='red'), name='최적점'
    ))
    fig.update_layout(title="목적 함수 3D 그래프",
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x,y)'))
    st.plotly_chart(fig, use_container_width=True)
