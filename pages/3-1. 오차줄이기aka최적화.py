import streamlit as st
import numpy as np
import plotly.express as px

st.set_page_config(page_title="계수를 조절하는 최적화의 본질", layout="wide")

st.title("🎯 회귀했더니 ~ ~ ~ ~")

st.image("assets/회귀.png", use_container_width=True)

st.markdown("""
## 수 많은 회귀 함수,  
### 복잡해 보여도 **회귀모델의 핵심 과정은 동일합니다.**  

> 📌 _오차가 최소가 되도록 모델의 **계수(parameter)** 를 조절하는 것_

아래에서 각 스텝을 **직접 조작**해 보면서  
“결국 다 계수를 만지면서 오차를 줄이는 게임”이라는 걸 체감해 봅시다.
""")

st.markdown("---")

# 공통 데이터 (1차 선형 + 잡음)
np.random.seed(0)
x = np.linspace(0, 10, 30)
noise = np.random.normal(0, 2, size=x.shape)
y_linear = 2 * x + 3 + noise

# =========================
# Step 1. 선형 회귀 (직접 a, b 조절)
# =========================
st.header("🔹 Step 1. 선형 회귀 (Linear Regression)")

st.markdown("""
모델: $p(x) = a x + b$  
- **조절하는 것**: 기울기 $a$, 절편 $b$  
- **목표**: 실제 $y$와 예측 $p(x)$ 사이의 오차(예: MSE)를 최소화
""")

with st.expander("👉 직선의 기울기와 절편을 직접 조절해 보기", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        a = st.slider("기울기 a", -1.0, 4.0, 2.0, 0.1)
        b = st.slider("절편 b", -2.0, 6.0, 3.0, 0.1)
    y_hat = a * x + b
    mse1 = np.mean((y_linear - y_hat) ** 2)

    st.write(f"📉 현재 MSE(평균제곱오차): **{mse1:.3f}**")

    fig1 = px.scatter(
        x=x,
        y=y_linear,
        labels={"x": "x", "y": "y"},
        title="데이터 vs 직선 모델"
    )
    fig1.add_scatter(x=x, y=y_hat, mode="lines", name="예측 직선")
    st.plotly_chart(fig1, use_container_width=True)

    st.caption("➡ 기울기와 절편을 바꾸면서, '오차가 가장 작아지는 조합'을 찾는 것이 바로 **최적화**입니다.")

st.markdown("---")

# =========================
# Step 2. 다항 회귀 (2차)
# =========================
st.header("🔹 Step 2. 다항 회귀 (Polynomial Regression, 2차)")

st.markdown("""
이번에는 **2차식**으로 가정해 봅니다.

모델:  $p(x) = a_2 x^2 + a_1 x + a_0$  

- 계수가 하나 더 생겨서 모양이 **곡선**이 됩니다.
- 그래도 여전히 하는 일은 **계수 $(a_2, a_1, a_0)$를 조절해 오차를 줄이는 것**입니다.
""")

with st.expander("👉 2차식 계수를 조절해 보기", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        a2 = st.slider("이차항 계수 a₂", -1.0, 1.0, 0.0, 0.05)
    with col2:
        a1 = st.slider("일차항 계수 a₁", 0.0, 4.0, 2.0, 0.1)
    with col3:
        a0 = st.slider("상수항 a₀", 0.0, 6.0, 3.0, 0.1)

    y_poly = a2 * x**2 + a1 * x + a0
    mse2 = np.mean((y_linear - y_poly) ** 2)
    st.write(f"📉 현재 MSE(평균제곱오차): **{mse2:.3f}**")

    fig2 = px.scatter(
        x=x,
        y=y_linear,
        labels={"x": "x", "y": "y"},
        title="데이터 vs 2차식 모델"
    )
    fig2.add_scatter(x=x, y=y_poly, mode="lines", name="2차식 예측 곡선")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("➡ 차수가 올라가고 항이 늘어날 뿐, 여전히 **계수를 조절해 오차를 줄이는 구조**입니다.")

st.markdown("---")

# =========================
# Step 3. 비선형 회귀 (exp 형태)
# =========================
st.header("🔹 Step 3. 비선형 회귀 (Nonlinear Regression)")

st.markdown("""
이번에는 지수함수 형태의 데이터를 상정해 봅니다.

모델 예시:  $p(x) = a e^{b x}$  

- 이제 $a, b$가 **지수 함수 안과 밖**에 들어가 있어서  
  오차 함수 모양도 비선형이 됩니다.
- 그래도 결국 **$a, b$를 조절해 오차를 줄이는 구조**는 같습니다.
""")

# 비선형 데이터
np.random.seed(2)
x_nl = np.linspace(0, 4, 40)
noise_nl = np.random.normal(0, 0.5, size=x_nl.shape)
y_nl_true = 2 * np.exp(0.8 * x_nl)
y_nl = y_nl_true + noise_nl

with st.expander("👉 a, b를 조절해 비선형 곡선을 맞춰 보기", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        a_nl = st.slider("계수 a", 0.0, 4.0, 2.0, 0.1)
    with col2:
        b_nl = st.slider("지수 계수 b", 0.0, 1.5, 0.8, 0.05)

    y_hat_nl = a_nl * np.exp(b_nl * x_nl)
    mse4 = np.mean((y_nl - y_hat_nl) ** 2)
    st.write(f"📉 현재 MSE(평균제곱오차): **{mse4:.3f}**")

    fig4 = px.scatter(
        x=x_nl,
        y=y_nl,
        labels={"x": "x", "y": "y"},
        title="비선형 데이터 vs 모델"
    )
    fig4.add_scatter(x=x_nl, y=y_hat_nl, mode="lines", name="비선형 예측 곡선")
    st.plotly_chart(fig4, use_container_width=True)

    st.caption("➡ 수식은 복잡해졌지만, 여전히 **'계수(a, b)를 조절해 오차를 줄이는' 최적화 문제**입니다.")

st.markdown("---")

# =========================
# Step 4. 다변수 회귀 (x1, x2 → y)
# =========================
st.header("🔹 Step 4. 다변수 회귀 (Multiple Regression)")

st.markdown("""
이번에는 입력 변수가 두 개인 상황을 가정해 봅니다.

모델:  $p(x_1, x_2) = w_1 x_1 + w_2 x_2 + b$

- 이제는 **여러 방향(축)**에서 오차를 줄여야 하기 때문에  
  파라미터 벡터 $(w_1, w_2, b)$를 동시에 조절합니다.
""")

# 예시 데이터 생성
np.random.seed(1)
n = 40
x1 = np.random.uniform(0, 5, n)
x2 = np.random.uniform(0, 5, n)
noise2 = np.random.normal(0, 1, n)
y_multi = 1.5 * x1 + 0.7 * x2 + 2 + noise2

with st.expander("👉 w₁, w₂, b를 조절하면서 여러 시각화로 보기", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        w1 = st.slider("w₁ (x1 계수)", 0.0, 3.0, 1.5, 0.1)
    with col2:
        w2 = st.slider("w₂ (x2 계수)", 0.0, 2.0, 0.7, 0.1)
    with col3:
        b_mv = st.slider("b (절편)", 0.0, 4.0, 2.0, 0.1)

    y_hat_multi = w1 * x1 + w2 * x2 + b_mv
    mse3 = np.mean((y_multi - y_hat_multi) ** 2)

    st.write(f"📉 현재 MSE(평균제곱오차): **{mse3:.3f}**")

    df_multi = {
        "x1": x1,
        "x2": x2,
        "y": y_multi,
        "y_hat": y_hat_multi,
        "오차": y_multi - y_hat_multi,
    }

    tab1, tab2, tab3, tab4 = st.tabs([
        "3D 산점도 (x1, x2, y)",
        "x1 vs y / y_hat",
        "x2 vs y / y_hat",
        "실제 y vs 예측 y_hat"
    ])

    with tab1:
        fig3d = px.scatter_3d(
            df_multi,
            x="x1",
            y="x2",
            z="y",
            color="오차",
            title="다변수 회귀: 데이터 (점)와 오차 색상"
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab2:
        fig_x1 = px.scatter(
            x=x1,
            y=y_multi,
            labels={"x": "x1", "y": "y"},
            title="x1 vs y, y_hat"
        )
        fig_x1.add_scatter(x=x1, y=y_hat_multi, mode="markers", name="예측 y_hat")
        st.plotly_chart(fig_x1, use_container_width=True)

    with tab3:
        fig_x2 = px.scatter(
            x=x2,
            y=y_multi,
            labels={"x": "x2", "y": "y"},
            title="x2 vs y, y_hat"
        )
        fig_x2.add_scatter(x=x2, y=y_hat_multi, mode="markers", name="예측 y_hat")
        st.plotly_chart(fig_x2, use_container_width=True)

    with tab4:
        fig_pred = px.scatter(
            x=y_multi,
            y=y_hat_multi,
            labels={"x": "실제 y", "y": "예측 y_hat"},
            title="실제값 vs 예측값"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    st.caption("➡ 여러 변수와 계수가 있어도, 여전히 하는 일은 **w₁, w₂, b를 잘 골라 오차를 줄이는 것**입니다.")

st.markdown("---")

# =========================
# Step 5. 수식 대신 쉬운 말로 정리
# =========================
st.header("🔹 Step 5. 최적화의 공통 구조 ")

st.markdown("""
In fact, 위 내용을 한 문장으로 정리하면 이렇게 말할 수 있습니다.

> **어떤 모양의 함수이든,  
> 000(손실)를 줄이기 위해  
> 그 함수의 000(파라미터)를 조절하는 과정.**

조금만 더 일반적으로 말하면:

1. **모델 선택**:  어떤 모양의 함수 $p(x; \\theta)$를 쓸지 정한다.  
2. **오차(손실) 정의**:  예측과 실제의 차이를 하나의 수 $L(\\theta)$로 정한다.  
3. **계수 조절**:  $L(\\theta)$가 줄어들도록 $\\theta$를 계속 바꿔본다.  
4. **멈추기**:  더 이상 눈에 띄게 줄지 않을 때, 그때의 $\\theta$를 "최적"이라고 부른다.

복잡한 AI 모델(딥러닝)도, 결국 이 네 줄 안에서 벗어나지 않습니다.
""")

with st.expander("힌트 보기 (000에 들어갈 말)", expanded=False):
    st.markdown("**예시**:  \n\n> 어떤 모양의 함수이든, **오차(손실)**를 줄이기 위해 그 함수의 **계수(파라미터)** 를 조절하는 과정.")

st.markdown("---")

st.success("정리: 선형이든, 다항이든, 다변수든, 비선형이든 결국 **'계수를 조절해서 오차를 줄이는 최적화'**라는 같은 틀 안에 있다.")
