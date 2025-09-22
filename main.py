import streamlit as st
import numpy as np
import altair as alt
import pandas as pd

st.title("🌱 정원 설계 비교 계산기")

st.markdown("같은 변수를 적용했을 때, 스파이럴·테라스·마운드·키홀 설계의 결과를 비교합니다.")

# --- 공통 변수 입력 ---
R = st.number_input("전체 반지름 R (m)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
H = st.number_input("최대 높이 H (m)", min_value=0.2, max_value=3.0, value=0.8, step=0.1)
alpha = st.slider("흙 대체율 α (0~1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
brick_len = st.number_input("벽돌 길이 (m)", min_value=0.1, max_value=1.0, value=0.2, step=0.05)

# 추가 변수
theta_max = st.number_input("스파이럴 회전각 θmax (라디안)", min_value=3.14, max_value=12.56, value=6.28, step=0.1)
n = st.number_input("테라스 층 수 n", min_value=1, max_value=10, value=3, step=1)
r_top = st.number_input("마운드 상단 반지름 r (m)", min_value=0.0, max_value=R, value=0.5, step=0.1)
theta_key = st.slider("키홀 통로 각도 θ (라디안)", min_value=0.1, max_value=3.14, value=1.57, step=0.1)

# --- 결과 저장용 ---
results = []

# 1) 스파이럴
theta = np.linspace(0, theta_max, 500)
r = (R/theta_max) * theta
dr_dtheta = R/theta_max
ds = np.sqrt(r**2 + dr_dtheta**2)
length = np.trapz(ds, theta)
bricks = length / brick_len
area = np.pi * R**2
volume = 2*np.pi*H*R**2/3
soil = volume * (1 - alpha)
results.append(["스파이럴", length, bricks, area, volume, soil])

# 2) 테라스
h_each = H/n
radii = np.linspace(0, R, n+1)
area, volume = 0, 0
for i in range(1, len(radii)):
    A = np.pi * (radii[i]**2 - radii[i-1]**2)
    area += A
    volume += A * h_each
length = 2*np.pi*R
bricks = length / brick_len
soil = volume * (1 - alpha)
results.append(["테라스", length, bricks, area, volume, soil])

# 3) 마운드
length = 2*np.pi*R
bricks = length / brick_len
area = np.pi * R**2
volume = (np.pi * H / 3) * (R**2 + R*r_top + r_top**2)
soil = volume * (1 - alpha)
results.append(["마운드", length, bricks, area, volume, soil])

# 4) 키홀
length = 2*np.pi*R
bricks = length / brick_len
area = np.pi * R**2 - 0.5 * R**2 * theta_key
volume = area * H
soil = volume * (1 - alpha)
results.append(["키홀", length, bricks, area, volume, soil])

# --- 데이터프레임 생성 ---
df = pd.DataFrame(results, columns=["설계안", "둘레(m)", "벽돌 수", "면적(㎡)", "부피(㎥)", f"흙 양(㎥, α={alpha})"])

# --- 표 출력 ---
st.subheader("📊 계산 결과 비교")
st.dataframe(df.style.format("{:.2f}", subset=["둘레(m)", "면적(㎡)", "부피(㎥)", f"흙 양(㎥, α={alpha})"]))

# --- 시각화 ---
st.subheader("📈 설계안별 비교 그래프")

for col in ["둘레(m)", "벽돌 수", "면적(㎡)", "부피(㎥)", f"흙 양(㎥, α={alpha})"]:
    chart = alt.Chart(df).mark_bar().encode(
        x="설계안:N",
        y=alt.Y(f"{col}:Q"),
        tooltip=["설계안", col]
    ).properties(title=col, width=400, height=300)
    st.altair_chart(chart, use_container_width=True)
