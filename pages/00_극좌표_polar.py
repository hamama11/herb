import streamlit as st

st.set_page_config(page_title="GeoGebra ")

st.title("🌀 극좌표 시각화")

# GeoGebra에서 생성한 공개 그래프 링크의 iframe 임베드
geogebra_app_url = "https://www.geogebra.org/m/gswxgwua"  

st.components.v1.html(
    f'<iframe src="{geogebra_app_url}" width="100%" height="600" style="border:1px solid #ccc;"></iframe>',
    height=620,
    scrolling=True
)

st.set_page_config(page_title="극좌표 GeoGebra 시각화👁️", layout="centered")

# 첫 번째 앱 ( 극좌표 길이)
st.subheader("📍 극좌표 길이 (r, θ)")
st.components.v1.html(
    '''
    <iframe src="https://www.geogebra.org/classic/tyeyhrce"
            width="100%" height="600" style="border:1px solid #ccc;"></iframe>
    ''',
    height=620
)
# 두 번째 앱 (극좌표 넓이)
st.subheader("📐 극좌표 넓이")
st.components.v1.html(
    '''
    <iframe src="https://www.geogebra.org/classic/v4vduefc"
            width="100%" height="600" style="border:1px solid #ccc;"></iframe>
    ''',
    height=620
)

st.markdown("---")
st.caption("※ 극좌표 (r, θ)를 직교좌표 (x, y)로 변환하여 시각화합니다.")
