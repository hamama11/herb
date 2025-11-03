import streamlit as st

def show():
    st.title("🌿 온실 관리 페이지")
    st.write("여기는 온실 상태를 관리하는 페이지입니다.")

st.set_page_config(page_title="GeoGebra ")

st.title("🌀 극좌표 란?(r, θ)")

st.components.v1.html(
    f'<iframe src="https://www.geogebra.org/classic/gswxgwua" width="100%" height="600" style="border:1px solid #ccc;"></iframe>',
    height=620,
    scrolling=True
)

st.set_page_config(page_title="극좌표 GeoGebra 시각화👁️", layout="centered")

# 첫 번째 앱 ( 극좌표 길이)
st.title("🪭 곡선 길이 by 극좌표")
st.components.v1.html(
    '''
    <iframe src="https://www.geogebra.org/classic/tyeyhrce"
            width="100%" height="600" style="border:1px solid #ccc;"></iframe>
    ''',
    height=620,
    scrolling=True
)
# 두 번째 앱 (극좌표 넓이)
st.title("📐 곡선 넓이 by 극좌표")
st.components.v1.html(
    '''
    <iframe src="https://www.geogebra.org/classic/v4vduefc"
            width="100%" height="600" style="border:1px solid #ccc;"></iframe>
    ''',
    height=620,
    scrolling=True
)
# 극좌표 PPt
st.title("This is 극좌표")
st.components.v1.html(
    '''
    <iframe src="https://gamma.app/embed/rm93extpoygc6dn" style="width: 100%; height: 450px" allow="fullscreen" title="극좌표 곡선의 길이와 넓이 공식 원리"></iframe>
    ''',
    height=620,
    scrolling=True
)

st.markdown("---")
st.caption("※ 극좌표 (r, θ)를 직교좌표 (x, y)로 변환하여 시각화합니다.")
