import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="극좌표 활동지 (Altair)", layout="centered")
st.title("🎨 극좌표 활동지: 면적·길이 (Altair 전용)")

st.markdown("""
이 활동지는 **극좌표 곡선** r=f(θ)의 **면적**과 **곡선 길이**를 실험하며 이해합니다.  
- **Ez Ver.**: 부채꼴 넓이·작은 길이 조각 개념을 클릭으로 확인  
- **Hard Ver.**: 수치 적분으로 공식을 검증
""")

# ---------------------------
# 1) 함수 선택 / 파라미터
# ---------------------------
st.header("1) r = f(θ) 선택")
preset = st.selectbox(
    "예시 또는 직접 입력",
    [
        "아르키메데스 나선: r = a + b*θ",
        "선형 나선: r = k*θ",
        "장미 곡선: r = c*np.sin(n*θ)",
        "원: r = R",
        "직접 입력"
    ],
)

colA, colB = st.columns(2)
theta_min = colA.number_input("θ 최소값", value=0.0)
theta_max = colB.number_input("θ 최대값", value=6.28)
samples = st.slider("샘플 개수(정밀도)", min_value=200, max_value=3000, value=800, step=100)

if preset == "직접 입력":
    func_str = st.text_input("f(θ) = ", "1 + 0.2*theta")
else:
    if preset.startswith("아르키메데스"):
        func_str = "a + b*theta"
    elif preset.startswith("선형 나선"):
        func_str = "k*theta"
    elif preset.startswith("장미 곡선"):
        func_str = "c*np.sin(n*theta)"
    elif preset.startswith("원"):
        func_str = "R + 0*theta"
    else:
        func_str = "1 + 0.2*theta"

st.caption(f"선택된 함수:  r(θ) = {func_str}")

# 파라미터 입력 + 설명
with st.expander("📌 파라미터 입력 (옆에 의미 참고)", expanded=False):
    st.markdown("""
    - **a**: 시작 반지름(초기 위치)  
    - **b**: 각도 증가 1 rad 당 반지름 증가량  
    - **c**: 장미 곡선의 진폭  
    - **k**: 선형 나선의 기울기  
    - **n**: 장미 곡선의 꽃잎 개수(짝수면 2n, 홀수면 n)  
    - **R**: 원의 반지름  
    """)
    col1, col2, col3 = st.columns(3)
    a = col1.number_input("a", value=0.0)
    b = col2.number_input("b", value=0.2)
    k = col3.number_input("k", value=0.3)
    c = col1.number_input("c", value=2.0)
    n = col2.number_input("n", value=3.0)
    R = col3.number_input("R", value=2.0)

# 안전한 eval 환경
SAFE_NS = {"np": np, "theta": None, "a": a, "b": b, "k": k, "c": c, "n": n, "R": R}

def f_theta(theta_arr):
    local_ns = SAFE_NS.copy()
    local_ns["theta"] = theta_arr
    return eval(func_str, {"__builtins__": {}}, local_ns)

if theta_max <= theta_min:
    st.error("θ 최대값은 최소값보다 커야 합니다.")
    st.stop()

theta = np.linspace(theta_min, theta_max, samples)
r = f_theta(theta)
x = r * np.cos(theta)
y = r * np.sin(theta)
df = pd.DataFrame({"theta": theta, "r": r, "x": x, "y": y})

# ---------------------------
# 2) Altair 차트
# ---------------------------
st.header("2) 곡선 그리기")
Rmax = float(np.nanmax(np.abs(r))) if np.all(np.isfinite(r)) else 1.0
if Rmax == 0: Rmax = 1.0
Rgrid = float(np.ceil(Rmax * 1.05))

curve = alt.Chart(df).mark_line(color="cyan").encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[-Rgrid, Rgrid])),
    y=alt.Y("y:Q", scale=alt.Scale(domain=[-Rgrid, Rgrid])),
    tooltip=[alt.Tooltip("theta:Q", format=".3f"),
             alt.Tooltip("r:Q", format=".3f")]
).properties(width=500, height=500, title="r=f(θ) 그래프")

st.altair_chart(curve.interactive(), use_container_width=True)

# ---------------------------
# 3) Clike 정답 보기
# ---------------------------
st.header("3) Clike 정답 보기")

st.markdown("👉 극좌표에서 면적과 길이의 **작은 조각**을 표현하는 공식을 떠올려 보세요.")

with st.expander("정답 보기 (dA 공식)"):
    st.latex(r"dA = \tfrac{1}{2} r^2 d\theta")

with st.expander("정답 보기 (ds 공식)"):
    st.latex(r"ds = \sqrt{ (r d\theta)^2 + (dr)^2 }")

# ---------------------------
# 4) Hard Ver — 수치적 검증
# ---------------------------
st.header("4) Hard ver 수치 검증 하기")

dr_dtheta = np.gradient(r, theta)
integrand_L = np.sqrt(r**2 + dr_dtheta**2)
area = 0.5 * np.trapz(r**2, theta)
length = np.trapz(integrand_L, theta)

col1, col2 = st.columns(2)
col1.metric("면적 A ≈", f"{area:.6f}")
col2.metric("길이 L ≈", f"{length:.6f}")

with st.expander("공식 다시 보기 (LaTeX)"):
    st.latex(r"A = \tfrac{1}{2}\int_{\alpha}^{\beta} r(\theta)^2 \, d\theta")
    st.latex(r"L = \int_{\alpha}^{\beta} \sqrt{ r(\theta)^2 + \left(\frac{dr}{d\theta}\right)^2 } \, d\theta")
