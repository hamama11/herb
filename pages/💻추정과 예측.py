# -*- coding: utf-8 -*-
"""
📐 Taylor n차 근사 & Newton Method Explorer

- 함수 f(x) 선택
- 한 점 a를 중심으로 한 테일러 n차 다항식 여러 개를 체크박스로 선택
- 뉴턴 방법으로 f'(x)=0 인 정지점(극값 후보) 찾기
- 테일러 중심점 a와 뉴턴 초기값 x0를 동일하게 사용

의존성: streamlit, numpy, pandas, altair
실행: streamlit run 최적화.py
"""

import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="📐 Taylor & Newton Explorer", layout="wide")
st.title("📐 테일러 다항식과 뉴턴 방법 탐구")


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
    """n차 도함수 근사 (1차 미분을 n번 반복하는 방식)"""
    if n == 0:
        return f(x)
    g = f
    for _ in range(n):
        g_prev = g

        def g_new(t, g_prev=g_prev):
            return d1(g_prev, t, h=h)

        g = g_new
    return g(x)


def taylor_poly_values(f, a, x_arr, n, h=1e-4):
    """
    테일러 n차 다항식 T_n(x)를 점 x_arr에서 평가한 값
    T_n(x) = Σ_{k=0}^n (f^{(k)}(a)/k!) (x-a)^k
    """
    vals = np.zeros_like(x_arr, dtype=float)
    for k in range(n + 1):
        deriv = derivative_n(f, a, n=k, h=h)
        vals += deriv * (x_arr - a) ** k / math.factorial(k)
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
        "테일러 중심점 a (접선·근사 시작 위치)",
        min_value=x_min + 1e-6,
        max_value=x_max - 1e-6,
        value=(x_min + x_max) / 2,
        step=0.1,
        help="이 점에서 접선(테일러 1차)과 테일러 n차를 만듭니다. 뉴턴 방법도 이 점에서 출발합니다.",
    )

    st.markdown("#### 테일러 차수 선택 (여러 개 가능)")
    selected_degrees = []
    default_checked = {1, 2}  # 처음에는 1차, 2차만 켜두기
    for n in range(1, 6):
        checked = st.checkbox(f"{n}차 다항식 보기", value=(n in default_checked))
        if checked:
            selected_degrees.append(n)

    if not selected_degrees:
        st.warning("적어도 하나의 테일러 차수를 선택해 주세요. (예: 1차)")
        # 그래도 코드가 돌아가도록, 강제로 1차 추가
        selected_degrees = [1]

    st.divider()
    st.subheader("뉴턴 방법 (정지점 탐색)")
    st.caption("초기값 x₀는 위에서 정한 테일러 중심점 a와 동일하게 사용합니다.")
    iters = st.number_input("최대 반복 횟수", min_value=1, max_value=30, value=10)
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

base = pd.DataFrame({"x": x, "f(x)": fx})

# 테일러 여러 차수에 대한 값 (long 형식)
taylor_records = []
for n in selected_degrees:
    vals_n = taylor_poly_values(f, a, x, n)
    for xi, yi in zip(x, vals_n):
        taylor_records.append({"x": xi, "y": yi, "degree": f"T_{n}(x)"})

df_taylor = pd.DataFrame(taylor_records) if taylor_records else None

fn_a = f(a)


# ------------------------------
# 그래프 1: 원함수 + 테일러 n차(들)
# ------------------------------
st.subheader("① f(x)와 여러 테일러 다항식 비교")

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

# 선택된 테일러 다항식들
if df_taylor is not None and not df_taylor.empty:
    layers.append(
        alt.Chart(df_taylor)
        .mark_line(strokeDash=[6, 3])
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("degree:N", title="테일러 다항식"),
            tooltip=["x:Q", "y:Q", "degree:N"],
        )
        .properties(title="테일러 다항식들")
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
    .properties(height=420, title="f(x)와 테일러 근사 비교")
)

st.caption(
    "• 1차는 접선(선형근사), 2차 이상은 곡선의 굽음까지 반영합니다.\n"
    "• 여러 차수를 동시에 켜고, 곡선이 어떻게 달라지는지 비교해 보세요."
)
st.altair_chart(chart_main, use_container_width=True)


# ------------------------------
# 뉴턴 방법: f'(x)=0 정지점 찾기 (x0 = a)
# ------------------------------
st.subheader("② 뉴턴 방법으로 f'(x)=0인 정지점 찾아보기 (x₀ = a)")

newton_rows = []
cur = float(a)  # 초기값 x0를 테일러 중심점 a와 동일하게

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
    st.info("반복 과정에서 수렴하지 않았습니다. 함수나 중심점 a를 바꿔 보세요.")

# ②-2 표 (그래프 아래)
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
with st.expander("📘 개념 정리 (선형근사·테일러·뉴턴·최적화 분리해서 보기)"):
    st.markdown(
        r"""
- **선형근사(Linear Approximation)**  
  - 한 점 `a` 근처에서 함수 `f(x)`를 그 점에서의 **접선**으로 바꿔서 생각하는 것  
  - 식: \(L(x) = f(a) + f'(a)(x-a)\)  
  - 그래프에서는 곡선 대신 직선 하나로 국소적인 모습을 보는 느낌

- **테일러 n차 근사(Taylor Polynomial)**  
  - 한 점 `a` 주변에서 함수 값을 **다항식**으로 근사  
  - \(T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k\)  
  - n=1일 때 **선형근사와 같은 식**, n이 커질수록 곡선 모양을 더 잘 따라가지만  
    실제 계산에서는 **고차에서 오차·발산**이 생길 수도 있음

- **뉴턴 방법(Newton's Method)**  
  - 어떤 방정식의 해를 수치적으로 찾는 기법  
  - 여기서는 **\(f'(x)=0\)** 을 풀어서 **기울기가 0인 점(정지점)** 을 찾는 데 사용  
  - 점진적으로 \(x_{n+1} = x_n - \dfrac{f'(x_n)}{f''(x_n)}\) 로 이동하며 해를 추정  
  - 이 점이 극대/극소/변곡인지는 **2차 미분 \(f''(x)\)** 로 다시 판단

- **최적화(Optimization)**  
  - 더 큰 개념: “함수의 최댓값/최솟값(또는 가장 좋은 값)을 찾는 문제 전체”  
  - 여러 방법이 있음 (경사하강법, 뉴턴법, 탐색법, 준-뉴턴법 등)  
  - **뉴턴 방법은 그 중 하나의 도구**일 뿐이고,  
    항상 최적해에 수렴하는 것도 아니며  
    초기값과 함수 모양에 따라 **발산하거나 엉뚱한 정지점으로 갈 수도 있음**

정리하면,

> 선형근사 = 테일러 1차  
> 테일러 n차 = 그보다 일반적인 국소 근사  
> 뉴턴 방법 = (주로) f'(x)=0을 풀기 위한 알고리즘  
> 최적화 = "가장 좋은 값"을 찾으려는 전체적인 문제, 그 중 한 도구로 뉴턴법을 사용할 수 있음
"""
    )
# ------------------------------
# 학습의 의미 블록
# ------------------------------
with st.expander("🧭 학습의 의미: 테일러·뉴턴·최적화"):
    st.markdown(
        """
| 단계 | 탐구 개념 | 핵심 아이디어 | 수학적 의미 | 사고 확장 |
|------|-----------|----------------|-------------|-----------|
| **1️⃣ 테일러 다항식** | 복잡한 함수도 한 점 근처에서는 단순하게 볼 수 있다. | 한 점 주변에서 그래프를 직선(1차), 곡선(2차)으로 근사함. | 도함수 정보를 활용하여 변화를 예측함. | 현실의 복잡한 현상을 단순 모델로 바꾸어 보는 능력 |
| **2️⃣ 뉴턴 방법** | 접선을 이용해 반복적으로 해를 찾아갈 수 있다. | 1차 근사식을 반복 적용해 근사값을 개선함. | 기울기(미분)가 ‘방향’을 알려줌. | 경사하강법, 최적화 알고리즘의 기본 아이디어로 연결 |
| **3️⃣ 최적화** | “가장 좋은 값(최댓값·최솟값)”을 수학으로 찾는 과정 | 함수의 정지점(f′=0)을 찾아 극대·극소를 판별함. | 함수의 구조를 분석하고 의사결정 기준을 세움 | 공학·경제·AI 등 현실 문제 해결로 확장 가능 |

---

> 테일러 다항식은 **“변화를 단순하게 이해하는 도구”**,  
> 뉴턴 방법은 **“그 단순함을 반복해 답을 찾아가는 과정”**,  
> 최적화는 **“더 나은 상태를 수학적으로 판단하는 사고”**이다.  
>
> 복잡한 현상을 수학적으로 단순화하고,  
> 그 단순함 속에서 **방향과 최적점을 찾는 것** —  
> 이것이 테일러와 뉴턴 탐구의 의미이다.
        """
    )
