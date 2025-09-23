import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="나선 대표 SIX with Altair", layout="centered")
st.title("🌿 나선 대표 SIX — 함수 선택 · 변수 설정 · 그래프 · 길이/면적")

# 내부 수치 근사용 해상도(고정)
SAMPLES = 1200

# ---------------------------
# 1) 나선 선택
# ---------------------------
spiral = st.selectbox(
    "나선을 선택하세요",
    [
        "1) 아르키메데스  r = a + b·θ",
        "2) 로그          r = a·e^{bθ}",
        "3) 페르마        r² = a²·θ",
        "4) 쌍곡선        r = a/θ",
        "5) 클리소이드(코르누)  x(s), y(s)",
        "6) 헬릭스(투영)  x = R cos t, y = R sin t",
    ],
)

# ---------------------------
# 2) 범위 슬라이더(θ, t, s)
# ---------------------------
if spiral.startswith(("1)", "2)", "3)")):
    theta_min, theta_max = st.slider("θ 범위", 0.0, 20.0, (0.0, 2*np.pi), 0.01)
    t = np.linspace(theta_min, theta_max, SAMPLES)
elif spiral.startswith("4)"):
    theta_min, theta_max = st.slider("θ 범위 (0 제외)", 0.01, 20.0, (0.2, 2*np.pi), 0.01)
    t = np.linspace(theta_min, theta_max, SAMPLES)
elif spiral.startswith("5)"):
    s_min, s_max = st.slider("s 범위", 0.0, 30.0, (0.0, 6.0), 0.01)
    t = np.linspace(s_min, s_max, SAMPLES)   # s
else:  # 6) Helix
    t_min, t_max = st.slider("t 범위", 0.0, 20.0, (0.0, 2*np.pi), 0.01)
    t = np.linspace(t_min, t_max, SAMPLES)

# ---------------------------
# 3) 나선별 필요한 변수만 슬라이더로
# ---------------------------
params = {}
if spiral.startswith("1)"):  # Archimedean
    st.markdown("**아르키메데스**: a=시작 반지름, b=각도 1 rad당 반지름 증가량")
    col1, col2 = st.columns(2)
    params["a"] = col1.slider("a", -5.0, 5.0, 0.0, 0.01)
    params["b"] = col2.slider("b", -2.0, 2.0, 0.2, 0.01)

elif spiral.startswith("2)"):  # Logarithmic
    st.markdown("**로그 나선**: a=스케일(초기 반지름), b=성장률(크면 급격히 퍼짐)")
    col1, col2 = st.columns(2)
    params["a"] = col1.slider("a", 0.01, 5.0, 1.0, 0.01)
    params["b"] = col2.slider("b", -1.0, 1.0, 0.15, 0.01)

elif spiral.startswith("3)"):  # Fermat
    st.markdown("**페르마**: r² = a²θ → a=스케일 (θ≥0 권장)")
    params["a"] = st.slider("a", 0.01, 5.0, 1.0, 0.01)

elif spiral.startswith("4)"):  # Hyperbolic
    st.markdown("**쌍곡선 나선**: r = a/θ (θ→0에서 특이점)")
    params["a"] = st.slider("a", 0.01, 10.0, 1.0, 0.01)

elif spiral.startswith("5)"):  # Clothoid
    st.markdown("**클리소이드(코르누)**: 곡률 κ(s) ∝ s → k=곡률 증가율")
    params["k"] = st.slider("k (곡률 증가율)", 0.01, 5.0, 1.0, 0.01)

else:  # 6) Helix (projected)
    st.markdown("**헬릭스(투영)**: 원통 투영 반지름 R")
    params["R"] = st.slider("R", 0.1, 10.0, 1.5, 0.01)

# ---------------------------
# 4) 좌표 생성 (x,y) & r(θ) 필요 시
# ---------------------------
mode = None
if spiral.startswith("1)"):
    mode = "polar"
    a, b = params["a"], params["b"]
    theta = t
    r = a + b*theta
    x, y = r*np.cos(theta), r*np.sin(theta)

elif spiral.startswith("2)"):
    mode = "polar"
    a, b = params["a"], params["b"]
    theta = t
    r = a * np.exp(b*theta)
    x, y = r*np.cos(theta), r*np.sin(theta)

elif spiral.startswith("3)"):
    mode = "polar"
    a = params["a"]
    theta = t
    r = a * np.sqrt(np.maximum(theta, 0.0))
    x, y = r*np.cos(theta), r*np.sin(theta)

elif spiral.startswith("4)"):
    mode = "polar"
    a = params["a"]
    theta = t
    r = a / theta
    x, y = r*np.cos(theta), r*np.sin(theta)

elif spiral.startswith("5)"):
    mode = "param"
    k = params["k"]
    s = t
    phi = 0.5 * k * s**2
    # 누적 적분(사다리꼴 근사)
    dx = np.cos(phi); dy = np.sin(phi)
    x = np.concatenate([[0], np.cumsum((dx[:-1] + dx[1:]) * 0.5 * (s[1:] - s[:-1]))])
    y = np.concatenate([[0], np.cumsum((dy[:-1] + dy[1:]) * 0.5 * (s[1:] - s[:-1]))])
    x, y = x[:SAMPLES], y[:SAMPLES]  # 길이 맞추기

else:
    mode = "param"
    R = params["R"]
    tt = t
    x = R*np.cos(tt); y = R*np.sin(tt)

df = pd.DataFrame({"t": t, "x": x, "y": y})
if mode == "polar":
    df["theta"] = theta
    df["r"] = r

# ---------------------------
# 5) Altair 차트
# ---------------------------
st.subheader("그래프")
Rmax = float(np.nanmax(np.hypot(x, y)))
Rlim = float(np.ceil(max(Rmax, 1.0) * 1.05))
chart = alt.Chart(df).mark_line().encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[-Rlim, Rlim])),
    y=alt.Y("y:Q", scale=alt.Scale(domain=[-Rlim, Rlim])),
    tooltip=[alt.Tooltip("t:Q", format=".3f")]
).properties(width=520, height=520, title=spiral)
st.altair_chart(chart.interactive(), use_container_width=True)

# ---------------------------
# 6) 길이 & 면적(가능한 경우)
# ---------------------------
def polyline_length(x, y):
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))

if mode == "polar":
    theta = df["theta"].to_numpy()
    r = df["r"].to_numpy()
    dr_dtheta = np.gradient(r, theta)
    L = np.trapz(np.sqrt(r**2 + dr_dtheta**2), theta)
    A = 0.5 * np.trapz(r**2, theta)
else:
    L = polyline_length(df["x"].to_numpy(), df["y"].to_numpy())
    A = None  # 클리소이드/헬릭스(투영)는 극좌표 면적 정의가 애매

col1, col2 = st.columns(2)
col1.metric("곡선 길이 L (근사)", f"{L:.6f}")
col2.metric("면적 A (극좌표 가능 시)", "—" if A is None else f"{A:.6f}")

with st.expander("계산 정의"):
    if mode == "polar":
        st.latex(r"L=\int_{\theta_0}^{\theta_1}\sqrt{r(\theta)^2+\left(\frac{dr}{d\theta}\right)^2}\,d\theta")
        st.latex(r"A=\tfrac12\int_{\theta_0}^{\theta_1}r(\theta)^2\,d\theta")
        st.caption("※ 쌍곡선 나선은 θ=0에서 특이점이 있으므로 θ₀>0에서 시작하세요.")
    else:
        st.write("- 길이: 평면 투영 다각선 근사 길이")
        st.write("- 면적: 이 페이지에서는 극좌표 r=f(θ) 형태에만 계산합니다.")
