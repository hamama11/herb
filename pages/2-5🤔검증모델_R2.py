import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="결정계수 r² 탐구 (데이터 업로드 포함)", layout="wide")

st.title("🎯 내가 고른 p(x)가 데이터를 얼마나 잘 설명할까? (r² 탐구 확장판)")

st.markdown(
    """
    이 앱에서는  
    1️⃣ 직접 데이터를 입력하거나 CSV 파일을 업로드하고  
    2️⃣ 모델 형태(직선, 이차식, 로그형)를 선택한 뒤  
    3️⃣ 계수를 조절하여 p(x)를 만들어보며  
    **결정계수 r²이 어떻게 변하는지 관찰**합니다.
    """
)

# ------------------------------------------------
# 1️⃣ 데이터 입력 / 업로드
# ------------------------------------------------
st.header("1️⃣ 데이터 준비: 직접 입력 또는 CSV 업로드")

uploaded_file = st.file_uploader("CSV 파일 업로드 (x, y 열 포함)", type=["csv"])

if uploaded_file:
    try:
        data_df = pd.read_csv(uploaded_file)
        st.success(f"파일 업로드 성공! ({len(data_df)}행)")
    except Exception as e:
        st.error(f"CSV 파일을 불러오는 중 오류 발생: {e}")
        st.stop()
else:
    st.info("CSV 파일을 업로드하지 않았다면 아래 예시 데이터를 사용하세요.")
    # 예시 데이터: 20개 점 (대략 포물선 형태)
    x_vals = np.linspace(1, 20, 20)
    y_vals = -0.05 * (x_vals - 10) ** 2 + 8  # 대략 위로 볼록 곡선
    example_df = pd.DataFrame(
        {
            "x": x_vals,
            "y": np.round(y_vals, 2),
        }
    )
    data_df = st.data_editor(
        example_df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )

data_df = data_df.dropna()
if "x" not in data_df.columns or "y" not in data_df.columns:
    st.error("데이터에는 반드시 'x'와 'y' 열이 포함되어야 합니다.")
    st.stop()
if len(data_df) < 3:
    st.error("최소 3개 이상의 데이터가 필요합니다.")
    st.stop()

x = data_df["x"].to_numpy(dtype=float)
y = data_df["y"].to_numpy(dtype=float)

# ------------------------------------------------
# 2️⃣ 모델 선택 및 계수 조절
# ------------------------------------------------
st.header("2️⃣ p(x) 형태 및 계수 선택")

model_type = st.radio(
    "모델 형태 선택",
    options=["선형 (ax + b)", "이차식 (ax² + bx + c)", "로그형 (a ln x + b)"],
    horizontal=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    a = st.slider("a", -5.0, 5.0, 1.0, 0.1)
with col_b:
    b = st.slider("b", -10.0, 10.0, 0.0, 0.5)
with col_c:
    c = st.slider("c (이차식 전용)", -10.0, 10.0, 0.0, 0.5)

if model_type == "선형 (ax + b)":
    def p(x_): return a * x_ + b
    latex_p = rf"p(x) = {a:.2f}x + {b:.2f}"
elif model_type == "이차식 (ax² + bx + c)":
    def p(x_): return a * x_**2 + b * x_ + c
    latex_p = rf"p(x) = {a:.2f}x^2 + {b:.2f}x + {c:.2f}"
else:
    def p(x_): return a * np.log(x_) + b
    latex_p = rf"p(x) = {a:.2f}\ln x + {b:.2f}"

st.latex(latex_p)

# ------------------------------------------------
# 3️⃣ (먼저) 그래프로 시각화: 데이터 vs p(x)
# ------------------------------------------------
y_hat = p(x)
y_mean = np.mean(y)

SST = np.sum((y - y_mean) ** 2)
SSE = np.sum((y - y_hat) ** 2)
R2 = 1 - SSE / SST if SST != 0 else np.nan

st.header("3️⃣ 데이터 vs p(x) 시각화")

x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
y_grid = p(x_grid)

fig, ax = plt.subplots()
ax.scatter(x, y, label="데이터", color="black")
ax.plot(x_grid, y_grid, label="p(x)", color="blue")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title(f"데이터와 p(x) 비교 (R² = {R2:.3f})")
st.pyplot(fig)

# ------------------------------------------------
# 4️⃣ (그 다음) 계산 과정과 r² 값 확인
# ------------------------------------------------
st.header("4️⃣ 계산 과정과 r² 값 확인")

calc_df = pd.DataFrame({
    "x": x,
    "y (실제)": y,
    "p(x) = ŷ (예측)": np.round(y_hat, 3),
    "잔차 r = y - ŷ": np.round(y - y_hat, 3),
    "(y - ŷ)²": np.round((y - y_hat)**2, 3),
    "(y - ȳ)²": np.round((y - y_mean)**2, 3),
})
st.dataframe(calc_df, use_container_width=True)

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.latex(r"SST = \sum (y_i - \bar{y})^2")
    st.metric(label="SST (총변동)", value=f"{SST:.3f}")
with col_s2:
    st.latex(r"SSE = \sum (y_i - \hat{y}_i)^2")
    st.metric(label="SSE (오차변동)", value=f"{SSE:.3f}")
with col_s3:
    st.latex(r"R^2 = 1 - \dfrac{SSE}{SST}")
    st.metric(label="R² (결정계수)", value=f"{R2:.4f}")

# ------------------------------------------------
# 5️⃣ 탐구 가이드
# ------------------------------------------------
st.markdown(
    """
    ### 💡 탐구 아이디어
    - 슬라이더를 조절하면서 **p(x)** 모양이 변할 때 **R² 값이 어떻게 바뀌는지** 확인해보세요.  
    - CSV로 더 많은 실험 데이터를 넣어보고, 데이터 개수가 많아질수록  
      **선형 / 이차 / 로그 모델 중 어떤 것이 더 적절해 보이는지** 비교해 보세요.  
    - R²이 높더라도 특정 구간에서 잔차가 한쪽으로 몰리면 → **모델의 한계**를 이야기해볼 수 있습니다.  
    """
)
