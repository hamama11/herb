import streamlit as st
import numpy as np
from scipy.optimize import minimize, differential_evolution
import plotly.graph_objects as go

st.title("🧮 다변수 최적화 시각화")

st.markdown("""
이 앱은 2변수 함수의 **최적점**을 찾는 과정을 시각적으로 보여줍니다.
- **Local 최적화**: 초기값에 따라 다른 최적점으로 수렴할 수 있음
- **Global 최적화**: 항상 전역 최적점으로 수렴
""")

# 목적 함수
def f(x, y):
    return (x-2)**2 + (y-3)**2 + np.sin(3*x)*np.sin(3*y)

# 변수 슬라이더
x_val = st.slider("x", -1.0, 5.0, 0.0, 0.1)
y_val = st.slider("y", -1.0, 5.0, 0.0, 0.1)
st.write(f"현재 함수값 f(x,y) = {f(x_val, y_val):.4f}")

# 3D 그래프
x = np.linspace(-1, 5, 50)
y = np.linspace(-1, 5, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x,y)'))
st.plotly_chart(fig, use_container_width=True)

# 최적화 버튼
opt_method = st.radio("최적화 방식 선택", ["Local", "Global"])
if st.button("최적화 수행"):
    if opt_method == "Local":
        res = minimize(lambda vars: f(vars[0], vars[1]), x0=[x_val, y_val])
        st.success(f"Local 최적점: x={res.x[0]:.4f}, y={res.x[1]:.4f}, f={res.fun:.4f}")
    else:
        res = differential_evolution(lambda vars: f(vars[0], vars[1]), bounds=[(-1,5),(-1,5)])
        st.success(f"Global 최적점: x={res.x[0]:.4f}, y={res.x[1]:.4f}, f={res.fun:.4f}")
