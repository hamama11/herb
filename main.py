import streamlit as st

st.set_page_config(page_title="Math Adventure!", layout="centered")

# 제목
st.title("🎒 지혜를 갈구하는 탐구자들이여, 그대들의 발걸음을 진심으로 환영하노라.")

# 귀여운 이미지 삽입 (예: 공개 라이선스 일러스트)
st.image(
    "https://cdn.pixabay.com/photo/2022/07/28/10/07/math-7348592_1280.png",
    caption="귀여운 수학자 캐릭터들 🧠",
    use_column_width=True
)

# 소개 문구
st.markdown("""
보이지 않는 질문 속에 답이 숨겨져 있나니, 호기심의 불꽃이 켜지는 순간 새로운 길이 열리리라. 🧮💡

👉 왼쪽 사이드바에서 원하는 활동을 골라볼까요?
""")

# 탐험 버튼
if st.button("🚀 지금 !"):
    st.markdown("새로운 여정을 열고 싶다면, 왼쪽 사이드바를 클릭하여 첫걸음을 내딛으시오.")

# 바닥글
st.markdown("---")
st.caption("© 2025 SM Adventure | Powered by HaMath + 귀여움 ✨")
