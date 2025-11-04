import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="최적화 알고리즘 비교", layout="wide")

st.title("🚶‍♂️🚗✈️ 서로 다른 길로 가는 최적화: 미분법 vs 경사상승 vs 무작위탐색")

st.markdown(
    """
    같은 함수 \\( f(x) \\)에 대해 **세 가지 최적화 방법**을 비교해 봅니다.

    1. **미분법(해석적 해)** – 수식을 이용해 극값을 한 번에 계산  
    2. **경사 상승(Gradient Ascent)** – 현재 위치의 기울기를 따라 조금씩 올라감  
    3. **무작위 탐색(Random Search)** – 여러 후보를 임의로 찍어보고 가장 높은 값을 선택  

    👉 세 방법 모두 *비슷한 최적점*을 찾지만,  
    **“어떤 경로로 가는지”**가 다르다는 걸 눈으로 보게 하는 게 목표입니다.
    """
)

# ------------------------------------------------
# 1️⃣ 사용할 함수 f(x) 설정 (온실/생장률 느낌의 2차 함수)
# ------------------------------------------------
st.header("1️⃣ 공통 함수 f(x) 선택하기")

st.markdown(
    """
    기본 함수는 아래와 같습니다 (온실 온도–생장률 곡선 같은 형태):

    \\[
    f(x) = ax^2 + bx + c \\quad (a < 0 \\text{이면 위로 볼록 → 최댓값 존재})
    \\]

    - x: 온도, 비료량, 광고비 등 **투입량**이라고 생각해도 좋습니다.  
    - f(x): 생장률, 수확량, 매출 등 **결과/효과**라고 생각할 수 있습니다.
    """
)

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    a = st.number_input("계수 a (x² 앞)", value=-1.0, step=0.1, format="%.2f")
with col_f2:
    b = st.number_input("계수 b (x 앞)", value=6.0, step=0.5, format="%.2f")
with col_f3:
    c = st.number_input("계수 c (상수항)", value=3.0, step=0.5, format="%.2f")

st.latex(rf"f(x) = {a}x^2 + {b}x + {c}")

# x 구간 설정
col_rng1, col_rng2 = st.columns(2)
with col_rng1:
    x_min = st.number_input("x 최소값", value=0.0, step=0.5)
with col_rng2:
    x_max = st.number_input("x 최대값", value=8.0, step=0.5)

if x_max <= x_min:
    st.error("x 최대값은 최소값보다 커야 합니다.")
    st.stop()

x_grid = np.linspace(x_min, x_max, 400)

def f(x):
    return a * x**2 + b * x + c

def f_prime(x):
    return 2 * a * x + b

y_grid = f(x_grid)

# 기본 그래프
fig0, ax0 = plt.subplots()
ax0.plot(x_grid, y_grid, label="f(x)")
ax0.set_xlabel("x")
ax0.set_ylabel("f(x)")
ax0.set_title("최적화를 적용할 대상 함수 f(x)")
ax0.legend()
st.pyplot(fig0)

st.markdown(
    """
    - 그래프를 보고, “어느 근처에서 값이 최대일까?” 직관적으로 먼저 생각해 보게 할 수 있습니다.  
    - 이제 이 **같은 함수**에 대해 세 가지 다른 방법으로 최적점을 찾아볼게요.
    """
)

# ------------------------------------------------
# 2️⃣ 방법 1: 미분법(해석적 해)
# ------------------------------------------------
st.header("2️⃣ 방법 1 – 미분법으로 한 번에 최적점 찾기")

st.markdown(
    """
    도함수:

    \\[
    f'(x) = 2ax + b
    \\]

    극값 조건 \\( f'(x) = 0 \\) 을 풀면,

    \\[
    x^* = -\\frac{b}{2a}
    \\]

    (단, \\( a < 0 \\) 이면 이 지점은 **최댓값**입니다.)
    """
)

analytic_x = None
analytic_y = None

if a == 0:
    st.warning("a = 0이면 2차 함수가 아니므로 미분법으로 최댓값을 정의하기 애매해집니다.")
else:
    analytic_x = -b / (2 * a)
    analytic_y = f(analytic_x)

    st.write(
        f"**미분법으로 얻은 극값 후보:**  \n"
        f"- x* = -b / (2a) ≈ **{analytic_x:.3f}**  \n"
        f"- f(x*) ≈ **{analytic_y:.3f}**"
    )

    fig1, ax1 = plt.subplots()
    ax1.plot(x_grid, y_grid, label="f(x)")
    ax1.axvline(analytic_x, linestyle="--", label="미분법 최적 x*")
    ax1.scatter([analytic_x], [analytic_y], color="red")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("미분법으로 찾은 최적점")
    ax1.legend()
    st.pyplot(fig1)

# ------------------------------------------------
# 3️⃣ 방법 2: 경사 상승(Gradient Ascent)
# ------------------------------------------------
st.header("3️⃣ 방법 2 – 경사 상승: 기울기를 따라 한 걸음씩 올라가기")

st.markdown(
    r"""
    경사 상승은 다음과 같은 규칙으로 x를 업데이트합니다.

    \[
    x_{k+1} = x_k + \eta f'(x_k)
    \]

    - \\( x_0 \\): 시작점(초기 추측)  
    - \\( \eta \\): 학습률(한 번에 얼마나 이동할지)  
    - \\( f'(x_k) \\): 현재 위치에서의 기울기  

    기울기가 양수면 오른쪽으로, 음수면 왼쪽으로 조금씩 이동하며  
    **점점 더 높은 곳으로 올라가는 방법**입니다.
    """
)

if a == 0:
    st.info("a = 0인 경우 경사 상승 예시는 넘어갑니다 (선형 함수는 내부 최댓값이 없음).")
else:
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        x0 = st.slider(
            "초기값 x₀",
            float(x_min),
            float(x_max),
            float((x_min + x_max) / 4),
            step=float((x_max - x_min) / 20),
        )
    with col_g2:
        lr = st.slider("학습률 η", 0.001, 0.5, 0.1, 0.001)
    with col_g3:
        n_steps = st.slider("반복 횟수", 1, 50, 15)

    xs_path = [x0]
    ys_path = [f(x0)]

    x_curr = x0
    for _ in range(n_steps):
        grad = f_prime(x_curr)
        x_curr = x_curr + lr * grad
        xs_path.append(x_curr)
        ys_path.append(f(x_curr))

    fig2, ax2 = plt.subplots()
    ax2.plot(x_grid, y_grid, label="f(x)")
    ax2.scatter(xs_path, ys_path, marker="o", label="경사 상승 자취")
    ax2.plot(xs_path, ys_path, linestyle="--", alpha=0.7)
    if analytic_x is not None:
        ax2.axvline(analytic_x, linestyle=":", label="미분법 x*")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    ax2.set_title("경사 상승으로 최적점 근처에 접근하는 과정")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(
        f"""
        - 시작점: \\( x_0 = {x0:.3f} \\)  
        - 마지막 위치: \\( x_{{\\text{{final}}}} ≈ {xs_path[-1]:.3f} \\),  
          \\( f(x_{{\\text{{final}}}}) ≈ {ys_path[-1]:.3f} \\)

        👉 미분법으로 구한 x*와 비교해 보세요.  
        - 학습률 η가 너무 크면 **튀거나 발산**할 수 있고,  
        - 너무 작으면 **천천히 수렴**합니다.  
        이를 바꿔보면서 “경로의 차이”를 학생들이 체험하게 할 수 있습니다.
        """
    )

# ------------------------------------------------
# 4️⃣ 방법 3: 무작위 탐색(Random Search)
# ------------------------------------------------
st.header("4️⃣ 방법 3 – 무작위 탐색: 여러 점을 찍어보고 최고점을 고르기")

st.markdown(
    """
    이번에는 **기울기를 쓰지 않고**,  
    단순히 구간 [x_min, x_max]에서 여러 점을 랜덤으로 찍어보고  
    그 중에서 **f(x)가 가장 큰 지점**을 고르는 방법입니다.

    - 알고리즘:
      1. 구간 안에서 N개의 x를 무작위로 뽑는다.  
      2. 각각에 대해 f(x)를 계산한다.  
      3. 그 중 가장 큰 f(x)를 가진 x를 선택한다.
    """
)

col_r1, col_r2 = st.columns(2)
with col_r1:
    n_random = st.slider("무작위 샘플 개수 N", 5, 200, 30, 5)
with col_r2:
    seed = st.number_input("랜덤 시드", value=0, step=1)

np.random.seed(int(seed))
rand_xs = np.random.uniform(x_min, x_max, size=n_random)
rand_ys = f(rand_xs)
best_idx = np.argmax(rand_ys)
rand_best_x = rand_xs[best_idx]
rand_best_y = rand_ys[best_idx]

fig3, ax3 = plt.subplots()
ax3.plot(x_grid, y_grid, label="f(x)")
ax3.scatter(rand_xs, rand_ys, alpha=0.5, label="무작위 샘플")
ax3.scatter([rand_best_x], [rand_best_y], color="red", s=80, label="무작위 탐색 최적점")
if analytic_x is not None:
    ax3.axvline(analytic_x, linestyle=":", label="미분법 x*")
ax3.set_xlabel("x")
ax3.set_ylabel("f(x)")
ax3.set_title("무작위 탐색으로 찾은 최적점")
ax3.legend()
st.pyplot(fig3)

st.markdown(
    f"""
    - 무작위로 찍은 점들 중 **가장 좋은 점**:  
      \\( x_{{\\text{{best}}}} ≈ {rand_best_x:.3f} \\),  
      \\( f(x_{{\\text{{best}}}}) ≈ {rand_best_y:.3f} \\)

    - N을 늘리면 **더 좋은 점을 찾을 확률이 증가**하지만,  
      그만큼 계산도 많이 필요합니다.  
    """
)

# ------------------------------------------------
# 5️⃣ 세 방법 결과 비교 정리
# ------------------------------------------------
st.header("5️⃣ 세 방법의 결과 한눈에 비교")

analytic_info = {
    "x": analytic_x if analytic_x is not None else None,
    "y": analytic_y if analytic_y is not None else None,
}

grad_info = {
    "x": xs_path[-1] if a != 0 else None,
    "y": ys_path[-1] if a != 0 else None,
}

rand_info = {
    "x": rand_best_x,
    "y": rand_best_y,
}

rows = []
rows.append(
    {
        "방법": "미분법 (해석적 해)",
        "x* (또는 근사)": None if analytic_info["x"] is None else round(analytic_info["x"], 3),
        "f(x*)": None if analytic_info["y"] is None else round(analytic_info["y"], 3),
        "설명": "수식을 직접 미분해 극값을 한 번에 계산",
    }
)
rows.append(
    {
        "방법": "경사 상승 (Gradient Ascent)",
        "x* (또는 근사)": None if grad_info["x"] is None else round(grad_info["x"], 3),
        "f(x*)": None if grad_info["y"] is None else round(grad_info["y"], 3),
        "설명": "기울기를 따라 여러 번 이동하며 점점 최적점 근처로 수렴",
    }
)
rows.append(
    {
        "방법": "무작위 탐색 (Random Search)",
        "x* (또는 근사)": round(rand_info["x"], 3),
        "f(x*)": round(rand_info["y"], 3),
        "설명": "구간 안을 마구 찍어본 뒤, 가장 좋은 점 하나를 선택",
    }
)

import pandas as pd
summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True)

st.markdown(
    """
    ### 🧭 정리

    - **미분법**: 이론적으로 가장 깔끔하고 빠르지만,  
      함수식 f(x)를 알고 있어야 하고, 미분 가능해야 합니다.
    - **경사 상승**: 함수식을 완전히 몰라도 (또는 고차원이어도)  
      “기울기만 알면” 반복적으로 최적점에 근접할 수 있습니다.  
      → 딥러닝에서 쓰는 **경사하강법**의 원리.
    - **무작위 탐색**: 기울기조차 몰라도 사용할 수 있지만,  
      좋은 해를 찾으려면 샘플 수가 많이 필요하고 비효율적일 수 있습니다.

    👉 수업에서 학생들에게

    > “결과만 비교하면 셋 다 비슷해 보일 수 있지만,  
    > **어떤 길(과정)** 을 선택했는지가 훨씬 중요하다.”

    는 메시지를 던지기 좋습니다.  
    - 온실 상황에서는 “실제 실험(무작위 탐색)”,  
    - 회귀모델을 세웠을 때는 “미분법/경사상승”의 차이로 연결해서 설명할 수 있어요.
    """
)
