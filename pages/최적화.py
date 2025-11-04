# -*- coding: utf-8 -*-
"""
📐 Linear Approx vs Taylor n-th vs Newton Method

- 함수 f(x) 선택
- 한 점 a를 중심으로 한 테일러 n차 다항식 시각화 (n 슬라이더)
- 뉴턴 방법으로 f'(x)=0인 정지점(극값 후보) 찾기
- 최적화와 뉴턴 방법의 관계를 개념 메모로 분리 설명

의존성: streamlit, numpy, pandas, altair
실행: streamlit run app_linear_taylor_opt.py
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="📐 Taylor & Newton Explorer", layout="wide")
st.title("📐 테일러 근사와 뉴턴 방법 탐구")


# ------------------------------
# 함수 정의와 도메인
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
        "domain": (0.1, 6.0),
    },
    "sqrt(x)": {
        "f": lambda x: np.sqrt(x),
        "domain": (0.0, 9.0),
    },
    "logistic 1/(1+e^-x)": {
        "f": lambda x: 1 / (1 + np.exp(-x)),
        "domain": (-6.0, 6.0),
    },
    "cubic x^3-3x": {
        "f": lambda x: x**3 - 3 * x,
        "domain": (-3.5, 3.5),
    },
}


# ------------------------------
# 수치 미분: 1차, 2차, n차
# ------------------------------
def d1(f, x, h=1e-4):
    """중심차분으로 1차 도함수 근사"""
    return (f(x + h) - f(x - h)) / (2 * h)


def d2(f, x, h=1e-4):
    """중심차분으로 2차 도함수 근사"""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


def derivative_n(f, x, n=1, h=1e-4):
    """n차 도함수 근사 (재귀적으로 1차 미분 반복)"""
    if n == 0:
        return f(x)
    g = f
    for _ in range(n):
        g_prev = g
        g = lambda t, g_prev=g_prev: d1(g_prev, t, h=h)
    return g(x)


def taylor_poly_values(f, a, x_arr, n, h=1e-4):
    """
    테일러 n차 다항식 T_n(x)를 점 x_arr에서 평가한 값 반환
    T_n(x) = Σ_{k=0}^n (f^{(k)}(a)/k!) (x-a)^k
    """
    vals = np.zeros_like(x_arr, dtype=float)
    for k in range(n + 1):
        deriv = derivative_n(f, a, n=k, h=h)
        vals += deriv * (x_arr - a) ** k / np.math.factorial(k)
    return vals


# ------------------------------
# 사이드바 설정
# ------------------------------
with st.sidebar:
    st.header("⚙️ 설정")

    fname = st.selectbox("함수 선택", list(FUNC_REGISTRY.keys()), index=5)
    f = FUNC_REGISTRY[fname]["f"]
    dom_min, dom_max = FUNC_REGISTRY[fname]["domain"]

    x_min, x_max = st.slider(
        "표시 구간 [min, max]",
        min_value=float(dom_min),
        max_value=float(dom_max),
        value=(float(dom_min), float(dom_max)),
        step=0.1,
    )

    a = st.slider(
        "테일러 중심점 a (접선이 스치는 x값)",
        min_value=x_min + 1e-6,
        max_value=x_max - 1e-6,
        value=(x_min + x_max) / 2,
        step=0.1,
    )

    taylor_n = st.slider(
        "테일러 다항식 차수 n",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="n=1이면 접선(선형근사), n이 커질수록 곡선 모양을 더 잘 따라가지만 오차도 같이 커질 수 있어요.",
    )

    st.divider()
    st.subheader("뉴턴 방법 (정지점 찾기)")
    x0 = st.slider(
        "초기값 x0",
        min_value=x_min,
        max_value=x_max,
        value=a,
        step=0.1,
        help="여기서부터 출발해서 f'(x)=0인 점을 찾아가요.",
    )
    iters = st.number_input(
        "최대 반복 횟수",
        min_value=1,
        max_value=30,
        value=10,
    )
    tol = st.number_input(
        "수렴 기준 |Δx| < ...",
        min_value=1e-10,
        max_value=1e-2,
        value=1e-5,
        format="%e",
    )


# ------------------------------
# 데이터 생성
# ------------------------------
x = np.linspace(x_min, x_max, 501)
fx = f(x)

# 테일러 n차 다항식 값
fn_a = f(a)
fp_a = d1(f, a)
taylor_vals = taylor_poly_values(f, a, x, taylor_n)

base = pd.DataFrame(
    {
        "x": x,
        "f(x)": fx,
        f"T_{taylor_n}(x)": taylor_vals,
    }
)

# ------------------------------
# 그래프 1: 원함수 + 테일러 n차
# ------------------------------
layers = []

# 원함수 f(x)
layers.append(
    alt.Chart(base)
    .mark_line()
    .encode(
        x=alt.X("x:Q", title="x"),
        y=alt.Y("f(x):Q", title="값"),
        tooltip=["x:Q", "f(x):Q"],
    )
    .properties(title="원함수 f(x)")
)

# 테일러 n차
layers.append(
    alt.Chart(base)
    .mark_line(strokeDash=[6, 3])
    .encode(
        x="x:Q",
        y=alt.Y(f"T_{taylor_n}(x):Q", title="값"),
        tooltip=["x:Q", f"T_{taylor_n}(x):Q"],
    )
    .properties(title=f"테일러 {taylor_n}차 다항식")
)

# 중심점 a 표시
pt_a = pd.DataFrame({"x": [a], "y": [fn_a]})
layers.append(
    alt.Chart(pt_a)
    .mark_point(size=120)
    .encode(x="x:Q", y="y:Q")
    .properties(title="중심점 a")
)

chart_main = (
    alt.layer(*layers)
    .resolve_scale(y="shared")
    .properties(height=420, title="f(x)와 테일러 n차 근사 비교")
)

st.subheader("① f(x)와 테일러 n차 근사 그래프")
st.caption(
    "• n=1일 때는 접선(선형근사)이고, n을 키우면 (이론적으로는) 곡선 모양을 더 잘 따라갑니다.\n"
    "• 여기서는 수치 미분으로 계수를 구하기 때문에, n이 너무 크면 오차가 커질 수 있어요."
)
st.altair_chart(chart_main, use_container_width=True)


# ------------------------------
# 뉴턴 방법: f'(x)=0 정지점 찾기
# ------------------------------
st.subheader("② 뉴턴 방법으로 f'(x)=0인 점(정지점) 찾아보기")

newton_rows = []
cur = float(x0)

for k in range(int(iters)):
    fp = d1(f, cur)
    fpp = d2(f, cur)
    if abs(fpp) < 1e-12:
        newton_rows.append(
            {"iter": k, "x": cur, "f'(x)": fp, "f''(x)": fpp, "Δx": np.nan}
        )
        break
    step = fp / fpp
    nxt = cur - step
    newton_rows.append(
        {"iter": k, "x": cur, "f'(x)": fp, "f''(x)": fpp, "Δx": -step}
    )
    if np.isnan(nxt) or np.isinf(nxt):
        break
    if abs(nxt - cur) < tol:
        cur = nxt
        break
    # 너무 멀리 튀면 중단
    if nxt < x_min - (x_max - x_min) or nxt > x_max + (x_max - x_min):
        break
    cur = nxt

newton_df = pd.DataFrame(newton_rows)
opt_x = newton_df["x"].iloc[-1] if not newton_df.empty else np.nan
opt_y = f(opt_x) if not np.isnan(opt_x) else np.nan

# ②-1 그래프(먼저)
if not newton_df.empty:
    it_pts = pd.DataFrame(
        {
            "x": newton_df["x"],
            "y": [f(v) for v in newton_df["x"]],
            "iter": newton_df["iter"],
        }
    )

    iter_layer = (
        alt.Chart(it_pts)
        .mark_line(point=True)
        .encode(
            x="x:Q",
            y="y:Q",
            tooltip=["iter:Q", "x:Q", "y:Q"],
        )
        .properties(title="뉴턴 반복 경로")
    )

    base_curve = (
        alt.Chart(base)
        .mark_line()
        .encode(x="x:Q", y="f(x):Q")
        .properties(title="f(x)")
    )

    if np.isfinite(opt_x):
        opt_df = pd.DataFrame({"x": [opt_x], "y": [opt_y]})
        opt_layer = alt.Chart(opt_df).mark_point(size=160).encode(
            x="x:Q", y="y:Q"
        )
        chart_newton = (
            alt.layer(base_curve, iter_layer, opt_layer).properties(height=320)
        )
    else:
        chart_newton = (
            alt.layer(base_curve, iter_layer).properties(height=320)
        )

    st.altair_chart(chart_newton, use_container_width=True)
else:
    st.info("반복 과정에서 수렴하지 않았습니다. 초기값 x0를 바꿔 보세요.")

# ②-2 표(그래프 아래로 이동)
if not newton_df.empty:
    st.markdown("**뉴턴 반복 값 표**")
    st.dataframe(newton_df, use_container_width=True)

    if np.isfinite(opt_x):
        fpp_star = d2(f, opt_x)
        if fpp_star > 0:
            nature = "극솟값(볼록 위의 최저점)"
        elif fpp_star < 0:
            nature = "극댓값(오목 위의 최고점)"
        else:
            nature = "정확한 성질 판별 어려움 (변곡점 가능성)"
        st.success(
            f"추정 정지점 x* ≈ {opt_x:.6f}, f(x*) ≈ {opt_y:.6f}  → 성질: {nature}"
        )
    else:
        st.info("정지점 성질을 판별할 수 없습니다.")
else:
    st.info("표시할 반복 값이 없습니다.")


# ------------------------------
# 개념 메모: 선형근사, 테일러, 뉴턴, 최적화
# ------------------------------
with st.expander("📘 개념 정리 (선형근사·테일러·뉴턴·최적화)"):
    st.markdown(
        r"""
- **선형근사(Linear Approximation)**  
  - 한 점 `a` 근처에서 함수 `f(x)`를 그 점에서의 **접선**으로 바꿔서 생각하는 것  
  - 식: \(L(x) = f(a) + f'(a)(x-a)\)  
  - 그래프에서는 곡선 대신 직선 하나로 국소적인 모습을 보는 느낌

- **테일러 n차 근사(Taylor Polynomial)**  
  - 한 점 `a` 주변에서 함수 값을 다항식으로 근사  
  - \(T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k\)  
  - n=1일 때 **선형근사와 같은 식**, n이 커질수록 곡선 모양을 더 잘 따라가지만  
    실제 계산에서는 **고차에서 오차·발산**이 생길 수도 있음

- **뉴턴 방법(Newton's Method)**  
  - 어떤 방정식의 해를 수치적으로 찾는 기법  
  - 여기서는 **\(f'(x)=0\)** 을 풀어서 **기울기가 0인 점(정지점)** 을 찾는 데 사용  
  - 점진적으로 \(x_{n+1} = x_n - \dfrac{f'(x_n)}{f''(x_n)}\) 로 이동하며 해를 추정  
  - 이 점이 극대/극소/변곡인지는 **2차 미분 \(f''(x)\)** 로 다시 판단

- **최적화(Optimization)와의 관계**  
  - 최적화는 더 넓은 개념:  
    - "함수의 최댓값/최솟값을 찾기"가 목표  
    - 방법은 매우 다양함 (경사하강법, 뉴턴법, 준-뉴턴법, 탐색법 등)  
  - 뉴턴 방법은 그중 **하나의 수치적 도구**일 뿐이고,  
    항상 최적해에 수렴하는 것도 아니며  
    초기값, 함수 모양에 따라 **발산하거나 엉뚱한 정지점으로 갈 수도 있음**

정리하면,

> 선형근사 = 테일러 1차  
> 테일러 n차 = 그보다 일반적인 국소 근사  
> 뉴턴 방법 = (주로) f'(x)=0을 풀기 위한 알고리즘  
> 최적화 = "가장 좋은 값"을 찾으려는 전체적인 문제, 그 중 한 도구로 뉴턴법을 사용할 수 있음
"""
    )
