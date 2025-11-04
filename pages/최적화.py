# -*- coding: utf-8 -*-
"""
Streamlit: Linear Approximation vs Taylor 1st vs Optimization (Newton)
- Altair 시각화로 곡선, 접선(=테일러 1차), 테일러 2차를 비교
- 뉴턴 방법으로 극값(정지점) 탐색 과정을 시각화

의존성: streamlit, numpy, pandas, altair
실행: streamlit run app_linear_taylor_opt.py
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="📐 접선Linear vs 테일러Taylor vs 최적화Optimization", layout="wide")
st.title("📐 Linear Approx vs Taylor 1st vs Optimization")

# ------------------------------
# Functions and helpers
# ------------------------------
FUNC_REGISTRY = {
    "sin(x)": {
        "f": lambda x: np.sin(x),
        "domain": (-6.0, 6.0),
    },
    "exp(x)": {
        "f": lambda x: np.exp(x),
        "domain": (-3.0, 2.0),
    },
    "log(x)": {
        "f": lambda x: np.log(x),
        "domain": (0.05, 6.0),
    },
    "sqrt(x)": {
        "f": lambda x: np.sqrt(x),
        "domain": (0.0, 9.0),
    },
    "logistic 1/(1+e^-x)": {
        "f": lambda x: 1/(1+np.exp(-x)),
        "domain": (-6.0, 6.0),
    },
    "cubic x^3-3x": {
        "f": lambda x: x**3 - 3*x,
        "domain": (-3.5, 3.5),
    },
}

def d1(f, x, h=1e-4):
    return (f(x+h) - f(x-h)) / (2*h)

def d2(f, x, h=1e-4):
    return (f(x+h) - 2*f(x) + f(x-h)) / (h**2)

# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.header("설정")
    fname = st.selectbox("함수 선택", list(FUNC_REGISTRY.keys()), index=5)
    f = FUNC_REGISTRY[fname]["f"]
    dom_min, dom_max = FUNC_REGISTRY[fname]["domain"]

    x_min, x_max = st.slider("표시 구간 [min, max]", min_value=float(dom_min), max_value=float(dom_max), value=(float(dom_min), float(dom_max)), step=0.1)
    a = st.slider("중심점 a (근사/접선)", min_value=x_min+1e-6, max_value=x_max-1e-6, value=(x_min+x_max)/2, step=0.1)

    show_t2 = st.checkbox("테일러 2차도 표시", value=True)

    st.divider()
    st.subheader("뉴턴 방법 (최적화)")
    x0 = st.slider("초기값 x0", min_value=x_min, max_value=x_max, value=a, step=0.1)
    iters = st.number_input("최대 반복 횟수", min_value=1, max_value=100, value=15)
    tol = st.number_input("수렴 허용오차 (|Δx|)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

# ------------------------------
# Build data
# ------------------------------
x = np.linspace(x_min, x_max, 501)
fx = f(x)

# Linear (Taylor 1st)
f_a = f(a)
fp_a = d1(f, a)
lin = f_a + fp_a*(x - a)

# Taylor 2nd (optional)
if show_t2:
    fpp_a = d2(f, a)
    t2 = f_a + fp_a*(x-a) + 0.5*fpp_a*(x-a)**2

# Newton optimization iterations: solve f'(x)=0
# (정지점을 찾음; 그 점이 극대/극소/변곡인지 여부는 f''로 판단)
newton_rows = []
cur = float(x0)
for k in range(int(iters)):
    fp = d1(f, cur)
    fpp = d2(f, cur)
    if abs(fpp) < 1e-12:
        # 해석적 불안정: 2차 미분이 너무 작으면 중단
        newton_rows.append({"iter": k, "x": cur, "f'(x)": fp, "f''(x)": fpp, "Δx": np.nan})
        break
    step = fp/fpp
    nxt = cur - step
    newton_rows.append({"iter": k, "x": cur, "f'(x)": fp, "f''(x)": fpp, "Δx": -step})
    if np.isnan(nxt) or np.isinf(nxt):
        break
    if abs(nxt - cur) < tol:
        cur = nxt
        break
    # 범위를 벗어나면 살짝 클리핑(시각화 편의)
    if nxt < x_min - (x_max-x_min) or nxt > x_max + (x_max-x_min):
        break
    cur = nxt

newton_df = pd.DataFrame(newton_rows)
opt_x = newton_df["x"].iloc[-1] if not newton_df.empty else np.nan
opt_y = f(opt_x) if not np.isnan(opt_x) else np.nan

# ------------------------------
# Charts
# ------------------------------
base = pd.DataFrame({"x": x, "f(x)": fx, "Linear/T1": lin})

layers = []
# Original curve
layers.append(
    alt.Chart(base).mark_line().encode(
        x=alt.X("x:Q", title="x"),
        y=alt.Y("f(x):Q", title="값"),
        tooltip=["x:Q", "f(x):Q"]
    ).properties(title="원함수 f(x)")
)

# Linear (Taylor 1st)
layers.append(
    alt.Chart(base).mark_line(strokeDash=[6,4]).encode(
        x="x:Q", y="Linear/T1:Q", tooltip=["x:Q", "Linear/T1:Q"]
    ).properties(title="접선(=테일러 1차)")
)

# Taylor 2nd (optional)
if show_t2:
    t2_df = base.copy()
    t2_df["T2"] = t2
    layers.append(
        alt.Chart(t2_df).mark_line(strokeDash=[2,2]).encode(
            x="x:Q", y="T2:Q", tooltip=["x:Q", "T2:Q"]
        ).properties(title="테일러 2차")
    )

# point at a
pt_a = pd.DataFrame({"x":[a], "y":[f_a]})
layers.append(
    alt.Chart(pt_a).mark_point(size=100).encode(x="x:Q", y="y:Q").properties(title="중심점 a")
)

chart_main = alt.layer(*layers).resolve_scale(y='shared').properties(height=420)

st.subheader("그래프 비교")
st.altair_chart(chart_main, use_container_width=True)

# ------------------------------
# Newton iteration trace
# ------------------------------
st.subheader("뉴턴 방법: f'(x)=0 정지점 찾기")

if not newton_df.empty:
    st.write("**반복 경로** (x 값이 어떻게 이동하는지):")
    st.dataframe(newton_df, use_container_width=True)

    # Iteration points on curve
    it_pts = pd.DataFrame({
        "x": newton_df["x"],
        "y": [f(v) for v in newton_df["x"]],
        "iter": newton_df["iter"],
    })

    iter_layer = alt.Chart(it_pts).mark_line(point=True).encode(
        x="x:Q", y="y:Q", tooltip=["iter:Q","x:Q","y:Q"]
    ).properties(title="뉴턴 반복 경로")

    # Optimum marker
    if np.isfinite(opt_x):
        opt_df = pd.DataFrame({"x":[opt_x], "y":[opt_y]})
        opt_layer = alt.Chart(opt_df).mark_point(size=160).encode(x="x:Q", y="y:Q")
        chart2 = alt.layer(
            alt.Chart(base).mark_line().encode(x="x:Q", y="f(x):Q"),
            iter_layer,
            opt_layer
        ).properties(height=320)
    else:
        chart2 = alt.layer(
            alt.Chart(base).mark_line().encode(x="x:Q", y="f(x):Q"),
            iter_layer
        ).properties(height=320)

    st.altair_chart(chart2, use_container_width=True)

    if np.isfinite(opt_x):
        fpp_star = d2(f, opt_x)
        nature = "극솟값" if fpp_star>0 else ("극댓값" if fpp_star<0 else "불확정/변곡")
        st.success(f"추정 정지점 x* ≈ {opt_x:.6f}, f(x*) ≈ {opt_y:.6f}  → 성질: {nature}")
    else:
        st.info("수렴하지 않았거나 도메인을 벗어났습니다. 초기값 x0를 바꿔 보세요.")
else:
    st.info("반복 정보가 없습니다. (초기값/함수를 조정해 보세요)")

# ------------------------------
# Notes
# ------------------------------
with st.expander("개념 메모"):
    st.markdown(
        """
        - **선형근사**: 한 점 `a` 근처에서 접선으로 함수값을 근사합니다. \(L(x)=f(a)+f'(a)(x-a)\)
        - **테일러 1차**: 테일러 급수의 1차만 취한 것으로 선형근사와 동일합니다.
        - **최적화(뉴턴)**: \(f'(x)=0\)을 만족하는 정지점을 찾기 위해 \(x_{n+1}=x_n-\frac{f'(x_n)}{f''(x_n)}\)을 반복합니다.
        - 2차 미분이 너무 작으면(=평평) 뉴턴법은 불안정할 수 있어요. 초기값을 조정해 보세요.
        """
    )
