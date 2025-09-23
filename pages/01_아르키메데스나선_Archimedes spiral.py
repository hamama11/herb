import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="극좌표 활동지 (Altair)", layout="centered")
st.title("🎨 극좌표 활동지: 면적·길이 (Altair 전용)")

st.markdown("""
이 활동지는 **극좌표 곡선** \(r=f(\\theta)\)의 **면적**과 **곡선 길이**를 실험하며 이해합니다.  
- **고2 Ver.**: 부채꼴 넓이·작은 길이 조각 개념을 빈칸으로 체크  
- **고3 Ver.**: 수치 적분으로 공식을 검증
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
    func_str = st.text_input("f(θ) = ", "1 + 0.2*theta")  # 예: 1 + 0.2*theta
    params_help = "a,b,c,k,n,R 등 파라미터가 있으면 아래에 직접 입력하세요."
else:
    if preset.startswith("아르키메데스"):
        func_str = "a + b*theta"
    elif preset.startswith("선형 나선"):
        func_str = "k*theta"
    elif preset.startswith("장미 곡선"):
        func_str = "c*np.sin(n*theta)"
    elif preset.startswith("원"):
        func_str = "R + 0*theta"  # 상수
    else:
        func_str = "1 + 0.2*theta"
    params_help = "필요한 파라미터만 사용됩니다."

st.caption(f"선택된 함수:  r(θ) = {func_str}")
st.info(params_help)

# 파라미터 입력
with st.expander("파라미터 입력 (필요한 것만 사용)", expanded=False):
    col1, col2, col3 = st.columns(3)
    a = col1.number_input("a", value=0.0)
    b = col2.number_input("b", value=0.2)
    k = col3.number_input("k", value=0.3)
    c = col1.number_input("c", value=2.0)
    n = col2.number_input("n", value=3.0)
    R = col3.number_input("R", value=2.0)

# 안전한 eval 환경(허용된 이름만)
SAFE_NS = {"np": np, "theta": None, "a": a, "b": b, "k": k, "c": c, "n": n, "R": R}

def f_theta(theta_arr):
    local_ns = SAFE_NS.copy()
    local_ns["theta"] = theta_arr
    return eval(func_str, {"__builtins__": {}}, local_ns)

# 파라미터 유효성
if theta_max <= theta_min:
    st.error("θ 최대값은 최소값보다 커야 합니다.")
    st.stop()

# ---------------------------
# 2) 데이터 생성
# ---------------------------
theta = np.linspace(theta_min, theta_max, samples)
try:
    r = f_theta(theta)
except Exception as e:
    st.error(f"함수 계산 중 오류: {e}")
    st.stop()

# 음수 r 처리(Polar 변환 시 해석 이슈) → 시각화만 위해 허용, 그대로 x,y 변환
x = r * np.cos(theta)
y = r * np.sin(theta)

df = pd.DataFrame({"theta": theta, "r": r, "x": x, "y": y})

# ---------------------------
# 3) 차트 (Altair: 직교좌표로 변환하여 표시)
#     + Polar 느낌의 안내선(원형 격자 & 방사선) 레이어
# ---------------------------
st.header("2) 곡선 그리기 (Altair)")
Rmax = float(np.nanmax(np.abs(r))) if np.all(np.isfinite(r)) else 1.0
if Rmax == 0: Rmax = 1.0
Rgrid = float(np.ceil(Rmax * 1.05))

# 원형 격자 데이터 (반지름 4~6개 정도)
rings = np.linspace(Rgrid/6, Rgrid, 6)
ring_df = pd.concat([
    pd.DataFrame({
        "x": ring*np.cos(np.linspace(0, 2*np.pi, 361)),
        "y": ring*np.sin(np.linspace(0, 2*np.pi, 361)),
        "ring": f"r = {ring:.2f}"
    })
    for ring in rings
])

# 방사선 안내선 (0, 30, 60, ... 도)
angles = np.deg2rad(np.arange(0, 180, 30))  # 반대편은 대칭으로 충분
ray_df = pd.concat([
    pd.DataFrame({"x": [0, Rgrid*np.cos(ang)], "y": [0, Rgrid*np.sin(ang)], "deg": f"{np.rad2deg(ang):.0f}°"})
    for ang in angles
])

base = alt.Chart().properties(width=520, height=520)

layer_rings = base.mark_line(opacity=0.18).encode(
    x="x:Q", y="y:Q", detail="ring:N"
).transform_fold(["x","y"])

# 위 transform_fold는 시연용이므로 간단히 레이어로 교체
layer_rings = alt.Chart(ring_df).mark_line(opacity=0.18).encode(
    x="x:Q", y="y:Q", detail="ring:N"
)

layer_rays = alt.Chart(ray_df).mark_line(opacity=0.18).encode(
    x="x:Q", y="y:Q", detail="deg:N"
)

curve = alt.Chart(df).mark_line().encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[-Rgrid, Rgrid])),
    y=alt.Y("y:Q", scale=alt.Scale(domain=[-Rgrid, Rgrid])),
    tooltip=[alt.Tooltip("theta:Q", format=".3f"),
             alt.Tooltip("r:Q", format=".3f")]
).properties(title="직교좌표로 본 r=f(θ)")

st.altair_chart((layer_rings + layer_rays + curve).interactive(), use_container_width=True)

# ---------------------------
# 4) 고2 활동: 빈칸 체크
# ---------------------------
st.header("3) 고2 활동 — 빈칸 채우기")

with st.form(key="g2"):
    a1 = st.text_input("① 작은 면적 조각 dA = ?", value="")
    a2 = st.text_input("② 작은 길이 조각 ds = ?", value="")
    submitted = st.form_submit_button("정답 확인")

    if submitted:
        ans1_ok = ("1/2" in a1 or "1/ 2" in a1 or "½" in a1) and ("r^2" in a1 or "r**2" in a1) and ("dθ" in a1 or "dtheta" in a1 or "d\\theta" in a1)
        ans2_ok = ("sqrt" in a2 or "√" in a2) and (("r" in a2 and "dθ" in a2) or ("rdθ" in a2) or ("r dθ" in a2)) and ("dr" in a2)

        st.write("① dA 정답 예시:  `1/2 * r^2 dθ`  또는  `½ r^2 dθ`")
        st.success("① 정답에 가깝습니다. 👍") if ans1_ok else st.error("① 핵심요소(½, r^2, dθ)를 모두 포함해야 합니다.")
        st.write("② ds 정답 예시:  `√[(r dθ)^2 + (dr)^2]`  또는  `sqrt((r*dθ)**2 + (dr)**2)`")
        st.success("② 정답에 가깝습니다. 👍") if ans2_ok else st.error("② 핵심요소(√, r dθ, dr)를 모두 포함해야 합니다.")

# ---------------------------
# 5) 고3 활동: 수치 적분으로 계산 검증
# ---------------------------
st.header("4) 고3 활동 — 수치적 검증")

# 수치 미분/적분
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

st.caption("※ Altair는 극좌표를 직접 지원하지 않아, x = r cosθ, y = r sinθ 로 변환하여 시각화합니다.")
