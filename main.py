import streamlit as st

# 제목
st.title("""🎒 지혜를 갈구하는 탐구자들이여,
그대들의 발걸음을 진심으로 환영하노라.""")

# 귀여운 이미지 삽입 (예: 공개 라이선스 일러스트)
st.image("assets/derpy_tiger.png",use_container_width=True)


# 소개 문구
st.markdown("""
보이지 않는 질문 속에 답이 숨겨져 있나니, 호기심의 불꽃이 켜지는 순간 새로운 길이 열리리라. 🧮💡
""")

# 탐험 버튼
if st.button("🐾지금🧠!"):
    st.markdown("새로운 여정을 열고 싶다면, 왼쪽 사이드바를 클릭하여 첫걸음을 내딛으시오.")

# 바닥글
st.markdown("---")
st.caption("© 2025 SM Adventure | Powered by HaMath + 귀여움 ✨")
