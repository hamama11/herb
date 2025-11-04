import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="데이터 기반 최적화 실험", layout="wide")

st.title("📈 함수를 모를 때, 데이터로 최적점 추론하기 (다항식·로그형·지수형 비교)")

st.markdown(
    """
    이 페이지는 **함수를 모를 때 데이터로 최적점을 추론하는 과정**을  
    여러 모델(다항식, 로그형, 지수형)로 비교하는 실험입니다.

    1. **데이터 점 찍기** (또는 예시 데이터 사용)  
    2. **모델 가정**: 다항식 / 로그형 / 지수형 선택  
    3. 각 모델로 **회귀(Regression)** → 근사 함수 \\( \\hat{f}(x) \\) 만들기  
    4. 다항식은 **미분으로 극값(최적점)** 계산, 로그·지수형은 **경계값 비교**  
    5. 그래프에 여러 모델의 **자취가 한꺼번에 남도록** 시각화
    """
)

# ------------------------------------------------
# 1️⃣ 데이터 입력 / 예시 데이터
# ------------------------------------------------
st.header("1️⃣ 데이터 점 찍기: x–y 관계 관찰하기")

st.markdown(
    """
    아래 표는 예시 데이터입니다.  
    - x: 입력 (예: 온실 온도, 광고비, 비료량 등)  
    - y: 결과 (예: 생장률, 매출, 수확량 등)  

    값을 직접 수정하거나 행을 추가/삭제해 보세요.
    """
)

example_df = pd.DataFrame(
    {
        "x": [1, 2, 3, 4, 5, 6],
        "y": [4.1, 6.2, 7.8, 8.2, 8.0, 7.1],
    }
)

data_df = st.data_editor(
    example_df,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor",
)

data_df = data_df.dropna()
if data_df.shape[0] < 3:
    st.error("최소 3개 이상의 데이터점이 필요합니다.")
    st.stop()

x = data_df["x"].to_numpy(dtype=float)
y = data_df["y"].to_numpy(dtype=float)

# ------------------------------------------------
# 2️⃣ 데이터 패턴 시각화 (탐색적 분석)
# ------------------------------------------------
st.subheader("2️⃣ 패턴 인식: 증가→감소 형태가 보이나?")

fig1, ax1 = plt.subplots()
ax1.scatter(x, y)
ax1.set_xlabel("x (입력)")
ax1.set_ylabel("y (결과)")
ax1.set_title("데이터 산점도")
st.pyplot(fig1)

st.markdown(
    """
    여기서는 수식 f(x)는 전혀 모릅니다.  
    다만 그래프를 보고,

    > “x가 커질수록 y는 계속 증가하는가?  
    > 어느 지점 이후에는 y가 줄어드는가?”

    같은 질문을 던지며 **형태를 추측(model hypothesis)** 하는 단계입니다.
    """
)

# ------------------------------------------------
# 3️⃣ 모델 유형 선택 (다항식 차수 체크박스 + 로그형 + 지수형)
# ---------
