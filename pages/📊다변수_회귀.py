# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("📊 다변수 회귀 탐구: 온실 데이터로 예측하기")

st.write("온도, 습도, 광량이 함께 잎 길이에 미치는 영향을 다변수 회귀로 탐구합니다.")

uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("샘플 데이터를 사용합니다.")
    df = pd.DataFrame({
        "평균온도":[23,25,26,27,28,26,24],
        "습도":[60,58,55,53,52,57,61],
        "광량":[18000,20000,24000,26000,27000,23000,19000],
        "잎길이":[4.1,4.6,5.2,5.7,6.1,5.8,5.0]
    })

st.dataframe(df)

# 모델 학습
X = df[["평균온도","습도","광량"]]
y = df["잎길이"]
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

st.subheader("📈 회귀식 결과")
coef_df = pd.DataFrame({
    "변수":["온도","습도","광량"],
    "기울기(β)":np.round(model.coef_,4)
})
st.table(coef_df)
st.write(f"**절편 β₀ = {model.intercept_:.3f}**, 결정계수 R² = {r2:.3f}")

# 시각화
st.subheader("📉 예측 vs 실제")
chart = alt.Chart(df).mark_circle(size=100).encode(
    x=alt.X("잎길이", title="실제 잎 길이(cm)"),
    y=alt.Y("예측값", title="예측 잎 길이(cm)")
).transform_calculate(
    예측값="datum.평균온도 * {} + datum.습도 * {} + datum.광량 * {} + {}".format(
        model.coef_[0], model.coef_[1], model.coef_[2], model.intercept_
    )
)
st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    ### 🧠 생각해보기
    - 어떤 변수가 잎 길이에 가장 큰 영향을 주었나요?  
    - 광량이 줄어들면 성장량은 어떻게 변할까요?  
    - 이 모델의 R² 값이 높을수록 무엇을 의미하나요?
    """
)
