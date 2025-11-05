# -*- coding: utf-8 -*-
"""
📊 Taylor Polynomial + Gradient Descent + Newton Method Explorer
----------------------------------------------------------
이 앱은 테일러 다항식 근사와 최적화 알고리즘(경사하강법, 뉴턴법)을
한 화면에서 비교하며 학습할 수 있도록 통합한 버전입니다.

실행:
    streamlit run app_optimization_explorer.py
"""

import math
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------
# ⚙️ Streamlit 설정
# -------------------------------------------------
st.set_page_config(page_title="📊 최적화 알고리즘 탐구", layout="wide")
st.title("📊 테일러 다항식 · 경사하강법 · 뉴턴 방법 통합 탐구")

# -------------------------------------------------
# 함수 레지스트리
# -------------------------------------------------
FUNC_REGISTRY = {
    "cubic f(x)=x³-3x": {
        "f": lambda x: x**3 - 3*x,
        "grad": lambda x: 3*x**2 - 3,
        "hess": lambda x: 6*x,
        "domain": (-3.5, 3.5),
    },
    "sin(x)": {
        "f": np.sin,
        "grad": np.cos,
        "hess": lambda x: -np.sin(x),
        "domain": (-6, 6),
    },
    "exp(x)": {
        "f": np.exp,
        "grad": np.exp,
        "hess": np.exp,
        "domain": (-3, 2),
    },
}

# -------------------------------------------------
# ⚙️ 사이드바 설정
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")

    fname = st.selectbox("함수 선택", list(FUNC_REGISTRY.keys()), index=0)
    f = FUNC_REGISTRY[fname]["f"]
    grad = FUNC_REGISTRY[fname]["grad"]
    hess = FUNC_REGISTRY[fname]["hess"]
    dom_min, dom_max = FUNC_REGISTRY[fname]["domain"]

    x_min, x_max = st.slider(
        "표시 구간 [min, max]",
        min_value=float(dom_min),
        max_value=float(dom_max),
        value=(float(dom_min), float(dom_max)),
        step=0.1,
    )

    a = st.slider(
        "테일러 중심점 a (접선·근사 시작 위치)",
        min_value=x_min + 1e-6,
        max_value=x_max - 1e-6,
        value=(x_min + x_max) / 2,
        step=0.1,
        help="이 점에서 테일러 근사와 최적화 알고리즘을 시작합니다.",
    )

    selected_degrees = []
    for n in range(1, 5):
        if st.checkbox(f"{n}차 테일러 다항식 보기", value=(n in {1, 2})):
            selected_degrees.append(n)

    st.divider()
    st.subheader("💡 최적화 설정")
    iterations = st.slider("반복 횟수", 5, 200, 30)
    lr = st.slider("학습률 (경사하강법)", 0.01, 1.0, 0.1)
    show_gd = st.checkbox("경사하강법 표시", True)
    show_newton = st.checkbox("뉴턴 방법 표시", True)


# -------------------------------------------------
# 🎯 테일러 다항식 계산 함수
# -------------------------------------------------
def derivative_n(f, x, n=1, h=1e-5):
    """n차 도함수 근사"""
    if n == 0:
        return f(x)
    g = f
    for _ in range(n):
        g_prev = g
        def g_new(t, g_prev=g_prev): return (g_prev(t+h) - g_prev(t-h)) / (2*h)
        g = g_new
    return g(x)

def taylor_poly(f, a, x_arr, n):
    vals = np.zeros_like(x_arr)
    for k in range(n + 1):
        deriv = derivative_n(f, a, n=k)
        vals += deriv * (x_arr - a)**k / math.factorial(k)
    return vals

# -------------------------------------------------
# 🎯 최적화 알고리즘 (1D)
# -------------------------------------------------
def run_1d_opt(x0, method):
    x = float(x0)
    hist = [x]
    for _ in range(iterations):
        g = grad(x)
        if method == "GD":
            x -= lr * g
        elif method == "Newton":
            h = hess(x)
            if abs(h) < 1e-12:
                break
            x -= g / h
        hist.append(x)
    return np.array(hist)

# -------------------------------------------------
# 📈 데이터 준비
# -------------------------------------------------
X = np.linspace(x_min, x_max, 400)
Y = f(X)
fn_a = f(a)

# -------------------------------------------------
# 🔹 그래프 1: f(x) + 테일러 다항식
# -------------------------------------------------
st.subheader("① f(x)와 테일러 근사 비교")

base = pd.DataFrame({"x": X, "f(x)": Y})
layers = [
    alt.Chart(base).mark_line().encode(x="x", y="f(x)").properties(title="f(x)")
]

# 테일러 근사 추가
for n in selected_degrees:
    Yn = taylor_poly(f, a, X, n)
    df = pd.DataFrame({"x": X, "y": Yn, "degree": f"T{n}(x)"})
    layers.append(
        alt.Chart(df)
        .mark_line(strokeDash=[6, 3])
        .encode(x="x", y="y", color="degree:N")
    )

# 중심점 표시
pt = pd.DataFrame({"x": [a], "y": [fn_a]})
layers.append(alt.Chart(pt).mark_point(size=120, color="red").encode(x="x", y="y"))

chart_taylor = alt.layer(*layers).properties(height=400)
st.altair_chart(chart_taylor, use_container_width=True)

# -------------------------------------------------
# 🔹 그래프 2: 최적화 알고리즘 시각화
# -------------------------------------------------
st.subheader("② 경사하강법 vs 뉴턴 방법 비교 (1D)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", name="f(x)"))

if show_gd:
    hist_gd = run_1d_opt(a, "GD")
    fig.add_trace(go.Scatter(
        x=hist_gd, y=f(hist_gd),
        mode="lines+markers",
        line=dict(color="red", width=3),
        name="경사하강법"
    ))

if show_newton:
    hist_n = run_1d_opt(a, "Newton")
    fig.add_trace(go.Scatter(
        x=hist_n, y=f(hist_n),
        mode="lines+markers",
        line=dict(color="blue", width=3),
        name="뉴턴 방법"
    ))

fig.update_layout(
    xaxis_title="x",
    yaxis_title="f(x)",
    width=800, height=400,
    legend=dict(x=0.02, y=0.98)
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 📘 개념 정리 블록
# -------------------------------------------------
with st.expander("📘 개념 요약: 테일러 · 경사하강법 · 뉴턴 · 최적화"):
    st.markdown(
        r"""
| 개념 | 수학적 정의 | 핵심 아이디어 | 알고리즘과의 관계 |
|------|--------------|----------------|----------------|
| **테일러 다항식** | \(T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k\) | 복잡한 함수를 국소적으로 단순화 | 1차 → 접선근사 (GD 기반), 2차 → 곡률 반영 (Newton 기반) |
| **경사하강법** | \(x_{k+1}=x_k-\eta f'(x_k)\) | 기울기 방향으로 조금씩 이동 | 테일러 1차 근사에 기반 |
| **뉴턴 방법** | \(x_{k+1}=x_k-\frac{f'(x_k)}{f''(x_k)}\) | 곡률(이차 정보)까지 고려 | 테일러 2차 근사에 기반 |
| **최적화** | \( \min_x f(x) \) | 가장 좋은(작은) 값을 찾는 과정 | GD, Newton은 모두 이를 위한 도구 |

> **요약:**  
> 테일러 다항식은 함수의 "국소 모델",  
> 경사하강법은 1차 모델로 이동,  
> 뉴턴법은 2차 모델로 점프.  
> 모두 "최적화"라는 하나의 목표를 향한 다른 접근이다.
"""
    )

# -------------------------------------------------
# 🧭 학습의 의미
# -------------------------------------------------
with st.expander("🧭 학습 확장: 수학적 사고의 흐름"):
    st.markdown(
        """
| 단계 | 탐구 개념 | 핵심 사고 | 실제 연결 |
|------|------------|------------|-----------|
| **① 테일러 다항식** | 복잡한 함수를 단순하게 근사 | 국소적인 모델링 사고 | 물리현상·AI 모델 근사 |
| **② 뉴턴 방법** | 접선을 반복적으로 이용해 해 탐색 | 수치적 추정, 반복 알고리즘 사고 | 방정식 풀이, 최적화, 머신러닝 |
| **③ 경사하강법** | 기울기만으로 하강 경로 탐색 | 최적화 방향 개념 | 딥러닝 학습의 핵심 원리 |
| **④ 최적화** | “가장 좋은 상태” 찾기 | 목적함수 최소화 | 공학, 경제, AI, 데이터 분석 |

> 테일러 → 뉴턴 → 경사하강법 → 최적화  
> 이 흐름은 "국소 근사 → 반복적 개선 → 전역적 판단"으로 이어지는  
> 수학적 사고의 진화 과정입니다.
"""
    )
