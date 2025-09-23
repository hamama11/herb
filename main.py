import streamlit as st

st.set_page_config(page_title="Math Adventure!", layout="centered")

# 제목
st.title("🎒 수학 탐험대에 오신 걸 환영합니다!")

# 귀여운 이미지 삽입 (예: 공개 라이선스 일러스트)
st.image(
    "https://cdn.pixabay.com/photo/2022/07/28/10/07/math-7348592_1280.png",
    caption="귀여운 수학자 캐릭터들 🧠",
    use_column_width=True
)

# 소개 문구
st.markdown("""
여기는 **수학을 사랑하는 친구들**이 모여서  
재미있고 신기한 **미적분/극좌표/MBTI 추천 활동** 등을  
함께 탐구하는 곳이에요! 🧮💡

👉 왼쪽 사이드바에서 원하는 활동을 골라볼까요?
""")

# 탐험 버튼
if st.button("🚀 지금 바로 탐험 시작!"):
    st.markdown("왼쪽 사이드바에서 원하는 페이지를 선택해 주세요 🧭")

# 바닥글
st.markdown("---")
st.caption("© 2025 Math Adventure | Powered by Streamlit + 귀여움 ✨")
