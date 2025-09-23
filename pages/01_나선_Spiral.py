import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="대표적 나선 활동지 (Altair)", layout="centered")
st.title("🌿 대표적 나선 6종: 함수 선택 · 변수 설정 · 그래프 · 길이 · 면적")

# ---------------------------
# 1) 나선 선택
# ---------------------------
spiral = st.selectbox(
    "나선을 선택하세요",
    [
        "1) 아르키메데스 나선  r = a + b·θ",
        "2) 로그 나선         r = a·e^{bθ}",
        "3) 페르마 나선       r^2 = a^2 θ",
        "4) 쌍곡선 나선       r = a/θ",
        "5) 클리소이드(코르누) x(s), y(s) (Fresnel 적분 근사)",
        "6) 헬릭스(투영)      x=a cos t, y=a sin t (z 무시)"
    ],
)

# ---------------------------
# 2) 공통 범위/샘플
# ---------------------------
col0, col1 = st.columns(2)
samples = col0.slider("샘플 개수(정밀도)", 300, 5000, 1200, 100)

# 도메인 입력 (θ 또는 t 또는 s)
if spiral.startswith(("1)", "2)", "3)")):
    # θ in [θ0, θ1]
    t0 = col1.number_input("θ 최소값", value=0.0)
    t1 = st.number_input("θ 최대값", value=6.283)  # 2π
elif spiral.startswith("4)"):
    # 쌍곡선 나선은 θ=0 특이점 → 양수로 시작
    t0 = col1.number_input("θ 최소값 (>0)", value=0.2)
    t1 = st.number_input("θ 최대값", value=6.283)
elif spiral.startswith("5)"):
    # 클리소이드: 매개변수 s in [0, s_max]
    t0 = col1.number_input("s 시작값", value=0.0)
    t1 = st.number_input("s 끝값", value=6.0)
else:
    # 헬릭스(투영): t in [t0, t1]
    t0 = col1.number_input("t 최소값", value=0.0)
    t1 = st.number_input("t 최대값", value=6.283)

if t1 <= t0:
    st.error("오른쪽 경계가 왼쪽 경계보다 커야 합니다.")
    st.stop()

# ---------------------------
# 3) 파라미터 입력
# ---------------------------
with st.expander("📌 파라미터(변수) 의미", expanded=False):
    st.markdown(
        """
- **a**: 시작 반지름/스케일(초기 크기)
- **b**: 각도 1 rad당 반지름 증가율(아르키메데스), 또는 성장률(로그)
- **c**: (이 코드에선 사용하지 않음)
- **k**: (클리소이드 곡률 증가율) 또는 기타 기울기/스케일에 사용될 수 있는 심볼
- **R**: 원형/헬릭스 반지름(여기선 헬릭스 투영 반지름)
- **주의**: 이 페이지는 상단 6개 나선에 맞춰 필요한 변수만 실제로 사용합니다.
        """
    )

colA, colB, colC = st.columns(3)
a = colA.number_input("a", value=1.0)
b = colB.number_input("b", value=0.2)
k = colC.number_input("k (클리소이드용)", value=1.0)
R = st.number_input("R (헬릭스 투영 반지름)", value=1.5)

# ---------------------------
# 4) 데이터 생성 (x,y) & r(θ) 필요 시
# ---------------------------
t = np.linspace(t0, t1, samples)

r = None
x = None
y = None
mode = None  # 'polar' or 'param'

if spiral.startswith("1)"):  # Archimedean
    mode = 'polar'
    theta = t
    r = a + b * theta
    x, y = r * np.cos(theta), r * np.sin(theta)

elif spiral.startswith("2)"):  # Logarithmic
    mode = 'polar'
    theta = t
    r = a * np.exp(b * theta)
    x, y = r * np.cos(theta), r * np.sin(theta)

elif spiral.startswith("3)"):  # Fermat
    mode = 'polar'
    theta = t
    # r^2 = a^2 theta → theta >= 0 가정
    r = a * np.sqrt(np.maximum(theta, 0.0))
    x, y = r * np.cos(theta), r * np.sin(theta)

elif spiral.startswith("4)"):  # Hyperbolic
    mode = 'polar'
    theta = t
    r = a / theta
    x, y = r * np.cos(theta), r * np.sin(theta)

elif spiral.startswith("5)"):  # Clothoid / Cornu
    mode = 'param'
    s = t
    # 방향각 φ(s) = (k/2) s^2  (표준형에서 상수 스케일 생략 가능)
    phi = 0.5 * k * s**2
    # x(s) = ∫ cos(phi(s)) ds, y(s) = ∫ sin(phi(s)) ds  (수치 적분 근사)
    # 누적 적분(사다리꼴 근사)
    dx = np.cos(phi)
    dy = np.sin(phi)
    x = np.concatenate([[0], np.cumsum((dx[:-1] + dx[1:]) * 0.5 * (s[1:] - s[:-1]))])
    y = np.concatenate([[0], np.cumsum((dy[:-1] + dy[1:]) * 0.5 * (s[1:] - s[:-1]))])
    # 시작점 보정
    x = x[:samples]
    y = y[:samples]

elif spiral.startswith("6)"):  # Helix (projected)
    mode = 'param'
    tt = t
    x = R * np.cos(tt)
    y = R * np.sin(tt)
    # z는 무시(평면 투영). 길이는 원호 길이로 계산됨(실제 3D 길이와 다름).

df = pd.DataFrame({"t": t, "x": x, "y": y})
if mode == 'polar':
    df["theta"] = t
    df["r"] = r

# ---------------------------
# 5) 그래프 (Altair)
# ---------------------------
st.subheader("그래프")
# 보기 좋은 범위
Rmax = np.nanmax(np.hypot(x, y))
Rlim = float(np.ceil(max(Rmax, 1.0) * 1.05))
chart = alt.Chart(df).mark_line().encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[-Rlim, Rlim])),
    y=alt.Y("y:Q", scale=alt.Scale(domain=[-Rlim, Rlim])),
    tooltip=[alt.Tooltip("t:Q", format=".3f")]
).properties(width=520, height=520, title=spiral)

st.altair_chart(chart.interactive(), use_container_width=True)

# ---------------------------
# 6) 길이 & 면적 계산
# ---------------------------
def polyline_length(x, y):
    return np.sum(np.hypot(np.diff(x), np.diff(y)))

if mode == 'polar':
    theta = df["theta"].to_numpy()
    r = df["r"].to_numpy()
    dr = np.gradient(r, theta)
    L = np.trapz(np.sqrt(r**2 + dr**2), theta)               # 곡선 길이
    A = 0.5 * np.trapz(r**2, theta)                          # 극좌표 면적
else:
    # 매개변수형: 평면 투영 길이(헬릭스는 원호 길이), 면적은 정의 곤란 → 표시만
    L = polyline_length(x, y)
    A = None

colL, colA = st.columns(2)
colL.metric("곡선 길이 L (수치 근사)", f"{L:.6f}")
colA.metric("면적 A (극좌표 가능 시)", "—" if A is None else f"{A:.6f}")

# 안내문
with st.expander("계산 정의 설명"):
    if mode == 'polar':
        st.latex(r"L = \int_{\theta_0}^{\theta_1} \sqrt{r(\theta)^2 + \left(\frac{dr}{d\theta}\right)^2}\, d\theta")
        st.latex(r"A = \tfrac12 \int_{\theta_0}^{\theta_1} r(\theta)^2\, d\theta")
        st.caption("※ 쌍곡선 나선은 θ=0에서 특이점이 있어 θ0>0에서 시작해야 합니다.")
    else:
        st.write("- **길이**: 평면 상 다각선 근사 길이(헬릭스는 3D가 아닌 투영 길이).")
        st.write("- **면적**: 극좌표 \(r=f(\\theta)\)가 아닌 곡선(클리소이드/헬릭스)은 일반적 면적 정의가 애매하여 계산하지 않습니다.")
