import streamlit as st
import altair as alt
import pandas as pd
import math

# Streamlit session_state를 사용한 자취 저장
if "path" not in st.session_state:
    st.session_state["path"] = []

st.title("🌀 극좌표 점 이동 시각화")
st.markdown("반지름 **r**과 각도 **θ(도)**를 조절하여 극좌표의 점을 이동시키고, 자취를 확인해보세요!")

# 사용자 입력
r = st.slider("반지름 r", 0.0, 10.0, 5.0, 0.1)
theta_deg = st.slider("각도 θ (도)", 0, 360, 90, 1)

# 극좌표 → 직교좌표 변환
theta_rad = math.radians(theta_deg)
x = r * math.cos(theta_rad)
y = r * math.sin(theta_rad)

# 자취 저장
if st.button("📍 점 찍기"):
    st.session_state["path"].append((x, y))

# 자취 초기화
if st.button("🔄 자취 초기화"):
    st.session_state["path"] = []

# 현재 점과 자취 데이터프레임 만들기
current_df = pd.DataFrame({"x": [x], "y": [y]})
path_df = pd.DataFrame(st.session_state["path"], columns=["x", "y"])

# 시각화
base = alt.Chart(current_df).mark_circle(size=200, color="red").encode(
    x=alt.X("x", scale=alt.Scale(domain=[-11, 11])),
    y=alt.Y("y", scale=alt.Scale(domain=[-11, 11])),
    tooltip=["x", "y"]
)

path = alt.Chart(path_df).mark_line(color="blue").encode(
    x="x", y="y"
)

st.altair_chart(path + base, use_container_width=True)

st.markdown("---")
st.caption("※ 극좌표 (r, θ)를 직교좌표 (x, y)로 변환하여 시각화합니다.")
