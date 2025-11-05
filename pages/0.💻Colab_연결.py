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

    # HTML 버튼으로 예쁘게 링크
    button_html = f"""
    <div style="text-align: center; margin: 20px 0;">
        <a href="{COLAB_URL}" target="_blank">
            <button style="
                background-color: #4CAF50; /* 초록색 */
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                border-radius: 8px;
                cursor: pointer;">
                🚀 Colab 열기
            </button>
        </a>
    </div>
    """
    st.markdown(button_html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "※ 브라우저 팝업 차단이 켜져 있으면 새 탭이 안 뜰 수 있어요. "
        "이 경우 아래 주소를 복사해서 직접 붙여 넣어도 됩니다.\n"
        f"{COLAB_URL}"
    )

if __name__ == "__main__":
    show()
