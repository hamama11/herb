# -*- coding: utf-8 -*-
import streamlit as st

def show():
    st.title("📎 Colab 실습 페이지로 이동")

    st.write(
        """
        이 페이지에서는 온실 데이터를 분석하는 **Google Colab 노트북**으로 이동합니다.  
        아래 버튼을 눌러 새 탭에서 Colab을 열어 주세요.
        """
    )

    # Google Colab 노트북 공유 URL
    COLAB_URL = "https://colab.research.google.com/drive/1XnXU-AoqZ1QKqOlb_B5KjRMyHYW7AGko?usp=sharing"

    st.link_button("🚀 Colab 열기", COLAB_URL)

    st.markdown("---")
    st.caption(
        "※ 브라우저 팝업 차단이 켜져 있으면 새 탭이 안 뜰 수 있어요. "
        "이 경우 주소를 복사해서 직접 붙여 넣어도 됩니다."
        ""https://colab.research.google.com/drive/1XnXU-AoqZ1QKqOlb_B5KjRMyHYW7AGko?usp=sharing""
        
    )

if __name__ == "__main__":
    show()
