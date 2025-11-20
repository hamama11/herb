# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # 3D 회귀면, 손실곡면에 필요

st.set_page_config(page_title="회귀로 미래를 예측해보기", layout="wide")

# 🌿 아이콘 버전 예시설명 함수
def 예시설명(생활, 온실):
    st.markdown(
        f"""
    <div style='padding:10px; border-radius:12px; background-color:#f8f9fa; margin-bottom:8px;'>
        <p>🏠 <b>생활</b> — {생활}</p>
        <p>🌿 <b>온실</b> — {온실}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# =========================
# 상단 레이아웃: 이미지 + 제목
# =========================
st.image("assets/회귀.png", use_container_width=True)

st.title("🎯 회귀로 미래를 예측해보기")

st.markdown("---")

st.markdown(
    """
## 미래를 예측한다? That is 회귀 함수  
### 복잡해 보여도 **회귀모델의 핵심 과정은 동일합니다.**  

> 📌 _오차가 최소가 되도록 모델의 **계수(parameter)** 를 조절하는 것_

아래 STEP을 **직접 조작**해 보면서  "예측값과 실제 값"을 관찰해 봅시다.
"""
)

# =========================
# 공통: 예제 데이터 만들기
# =========================

# Step 1 데이터: 선형 패턴
np.random.seed(0)
x1_data = np.linspace(0, 10, 30)
noise1 = np.random.normal(0, 1.5, size=x1_data.shape)
y1_data = 1.8 * x1_data + 2 + noise1   # 진짜 계수: 1.8, 2
df_step1 = pd.DataFrame({"x": x1_data, "y": y1_data})

# Step 2 데이터: 2차 곡선 패턴
np.random.seed(1)
x2_data = np.linspace(-5, 5, 40)
noise2 = np.random.normal(0, 3.0, size=x2_data.shape)
# 진짜 계수: a2=0.4, a1=-2, a0=3
y2_data = 0.4 * x2_data**2 - 2 * x2_data + 3 + noise2
df_step2 = pd.DataFrame({"x": x2_data, "y": y2_data})

# Step 3 데이터: 지수 증가 비선형 패턴
np.random.seed(2)
x3_data = np.linspace(0, 4, 40)
noise3 = np.random.normal(0, 1.0, size=x3_data.shape)
# 진짜 계수: a=2, b=0.7
y3_data = 2 * np.exp(0.7 * x3_data) + noise3
df_step3 = pd.DataFrame({"x": x3_data, "y": y3_data})

# Step 4 데이터: 다변수 평면 패턴
np.random.seed(3)
n4 = 50
x4_1 = np.random.uniform(0, 5, n4)
x4_2 = np.random.uniform(0, 5, n4)
noise4 = np.random.normal(0, 1.0, n4)
# 진짜 계수: w1=1.2, w2=-0.8, b=5
y4_data = 1.2 * x4_1 - 0.8 * x4_2 + 5 + noise4
df_step4 = pd.DataFrame({"x1": x4_1, "x2": x4_2, "y": y4_data})

# =========================
# Step 1. 선형 회귀
# =========================
st.header("🔹 Step 1. 선형 회귀 (Linear Regression)")

st.markdown(
    """
모델: $p(x) = a x + b$  

- 데이터는 **대체로 직선 모양**입니다.
- **조절하는 것**: 기울기 $a$, 절편 $b$  
- **목표**: 실제 $y$와 예측 $p(x)$ 사이의 오차(예: MSE)를 최소화
"""
)

예시설명(
    "거의 직선처럼 보이는, 비례에 가까운 상황 📚",
    "비료량이 많을수록 잎 길이가 일정 비율로 증가하는 직선 패턴 🌿",
)

with st.expander("👉 직선의 기울기와 절편을 직접 조절해 보기", expanded=True):
    col_ctrl, _ = st.columns([1, 2])  # 슬라이더는 좁게
    with col_ctrl:
        a = st.slider("기울기 a", -1.0, 4.0, 1.8, 0.1)
        b = st.slider("절편 b", -2.0, 8.0, 2.0, 0.1)

    y1_hat = a * x1_data + b
    mse1 = np.mean((y1_data - y1_hat) ** 2)
    st.write(f"📉 현재 MSE(평균제곱오차): **{mse1:.3f}**")

    # ✅ 표 왼쪽, 그래프 오른쪽
    col_table, col_fig = st.columns([1, 2])
    with col_table:
        st.markdown("**데이터 표**")
        st.dataframe(df_step1, use_container_width=True, height=380)

    with col_fig:
        fig1 = px.scatter(
            x=x1_data,
            y=y1_data,
            labels={"x": "x", "y": "y"},
            title="Step 1: 선형 데이터 vs 직선 모델",
        )
        fig1.add_scatter(x=x1_data, y=y1_hat, mode="lines", name="예측 직선")
        fig1.update_traces(marker=dict(size=6))
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    st.caption(
        "➡ 기울기와 절편을 바꾸면서, '오차가 가장 작아지는 조합'을 찾는 것이 바로 **최적화**입니다."
    )

st.markdown("---")

# =========================
# Step 2. 다항 회귀 (2차)
# =========================
st.header("🔹 Step 2. 다항 회귀 (Polynomial Regression, 2차)")

st.markdown(
    """
이번에는 **U자 모양(포물선)**에 더 잘 맞는 데이터를 사용합니다.

모델:  $p(x) = a_2 x^2 + a_1 x + a_0$  

- 데이터는 **대체로 2차 곡선** 형태입니다.
- 하지만 여전히 하는 일은 **계수 $(a_2, a_1, a_0)$를 조절해 오차를 줄이는 것**입니다.
"""
)

st.caption( "산/컵 모양 → 어디선가 가장 좋은 지점(최적점)이 있는 상황")

예시설명(
    "운동량이 너무 적거나 많으면 컨디션이 떨어지고, 적당할 때 최고 🏃‍♀️",
    "온도가 너무 낮거나 높으면 성장 느리고, 중간 적정 온도에서 성장률이 최대가 되는 상황 ☀️",
)

with st.expander("👉 2차식 계수를 조절해 보기", expanded=False):
    col_ctrl = st.columns(3)
    with col_ctrl[0]:
        a2 = st.slider("이차항 계수 a₂", -1.0, 1.0, 0.4, 0.05)
    with col_ctrl[1]:
        a1 = st.slider("일차항 계수 a₁", -4.0, 4.0, -2.0, 0.1)
    with col_ctrl[2]:
        a0 = st.slider("상수항 a₀", -2.0, 8.0, 3.0, 0.1)

    y2_hat = a2 * x2_data**2 + a1 * x2_data + a0
    mse2 = np.mean((y2_data - y2_hat) ** 2)
    st.write(f"📉 현재 MSE(평균제곱오차): **{mse2:.3f}**")

    # ✅ 표 왼쪽, 그래프 오른쪽
    col_table, col_fig = st.columns([1, 2])
    with col_table:
        st.markdown("**데이터 표**")
        st.dataframe(df_step2, use_container_width=True, height=380)

    with col_fig:
        fig2 = px.scatter(
            x=x2_data,
            y=y2_data,
            labels={"x": "x", "y": "y"},
            title="Step 2: 포물선 데이터 vs 2차식 모델",
        )
        fig2.add_scatter(
            x=np.sort(x2_data),
            y=y2_hat[np.argsort(x2_data)],
            mode="lines",
            name="2차식 예측 곡선",
        )
        fig2.update_traces(marker=dict(size=6))
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.caption("➡ 직선을 쓰면 어색했던 데이터가, 2차식으로는 훨씬 잘 맞을 수 있습니다.")

st.markdown("---")

# =========================
# Step 3. 비선형 회귀 (exp)
# =========================
st.header("🔹 Step 3. 비선형 회귀 (Nonlinear Regression)")

st.markdown(
    """
이번에는 **지수적으로 증가하는 데이터**를 사용합니다.

모델 예시:  $p(x) = a e^{b x}$  

- 이제 $a, b$가 **지수 함수 안과 밖**에 들어가 있어서  
  오차 함수 모양도 비선형이 됩니다.
- 그래도 결국 **$a, b$를 조절해 오차를 줄이는 구조**는 같습니다.
"""
)
st.caption( "점점 더 빨라지는 증가 → 초반엔 느리지만 나중에 폭발하는 상황")

예시설명(
    "점점 더 빨라지는 증가 — 초반에는 변화가 느리다가 시간이 지날수록 급격히 늘어나는 상황 (예: 복리 저축) 💰",
    "초기에는 성장 느리다가 어느 순간 이후 세포 분열과 병균·곰팡이 확산이 급격히 빨라지는 성장 단계 🌱",
)

with st.expander("👉 a, b를 조절해 비선형 곡선을 맞춰 보기", expanded=False):
    col_ctrl = st.columns(2)
    with col_ctrl[0]:
        a3 = st.slider("계수 a", 0.0, 4.0, 2.0, 0.1)
    with col_ctrl[1]:
        b3 = st.slider("지수 계수 b", 0.0, 1.5, 0.7, 0.05)

    y3_hat = a3 * np.exp(b3 * x3_data)
    mse3 = np.mean((y3_data - y3_hat) ** 2)
    st.write(f"📉 현재 MSE(평균제곱오차): **{mse3:.3f}**")

    # ✅ 표 왼쪽, 그래프 오른쪽
    col_table, col_fig = st.columns([1, 2])
    with col_table:
        st.markdown("**데이터 표**")
        st.dataframe(df_step3, use_container_width=True, height=380)

    with col_fig:
        fig3 = px.scatter(
            x=x3_data,
            y=y3_data,
            labels={"x": "x", "y": "y"},
            title="Step 3: 지수형 데이터 vs 비선형 모델",
        )
        fig3.add_scatter(x=x3_data, y=y3_hat, mode="lines", name="비선형 예측 곡선")
        fig3.update_traces(marker=dict(size=6))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    st.caption("➡ 수식 모양만 달라졌을 뿐, 여전히 **a, b를 조절해 오차를 줄이는 최적화 문제**입니다.")

st.markdown("---")

# =========================
# Step 4. 다변수 회귀 (x1, x2 → y) + 손실곡면
# =========================
st.header("🔹 Step 4. 다변수 회귀 & 손실함수 (Multiple Regression & Loss Surface)")

st.markdown(
    """
이번에는 입력 변수가 **두 개(x1, x2)**인 상황입니다.

모델:  $p(x_1, x_2) = w_1 x_1 + w_2 x_2 + b$

- 데이터는 대체로 **한 평면 위에 흩어져 있는 모양**입니다.
- 이제는 **여러 방향(축)**에서 오차를 줄여야 하기 때문에  
  파라미터 벡터 $(w_1, w_2, b)$를 동시에 조절합니다.
"""
)

예시설명(
    "조건이 두 개 이상 동시에 작용하는 상황 — 공부시간은 점수를 올리고, 수면 부족은 점수를 깎는 복합 영향 😴📖",
    "광량은 성장에 +, 과습은 – 방향으로 작용하는 두 변수의 평면 관계 🌞💧",
)

with st.expander(
    "👉 w₁, w₂, b를 조절하면서 데이터공간 & 파라미터공간 동시에 보기", expanded=False
):
    col_ctrl = st.columns(3)
    with col_ctrl[0]:
        w1 = st.slider("w₁ (x1 계수)", -1.0, 3.0, 1.2, 0.1)
    with col_ctrl[1]:
        w2 = st.slider("w₂ (x2 계수)", -2.0, 2.0, -0.8, 0.1)
    with col_ctrl[2]:
        b4 = st.slider("b (절편)", 0.0, 10.0, 5.0, 0.1)

    # 현재 파라미터에서의 예측과 MSE
    y4_hat = w1 * x4_1 + w2 * x4_2 + b4
    mse4 = float(np.mean((y4_data - y4_hat) ** 2))
    st.write(f"📉 현재 MSE(평균제곱오차): **{mse4:.3f}**")

    df_step4_view = df_step4.copy()
    df_step4_view["y_hat"] = y4_hat
    df_step4_view["오차"] = df_step4_view["y"] - df_step4_view["y_hat"]

    # 손실곡면과 히트맵에 쓸 (w1, w2) 그리드와 MSE 계산 (파라미터 공간)
    w1_grid = np.linspace(-1.0, 3.0, 50)
    w2_grid = np.linspace(-2.0, 2.0, 50)
    W1, W2 = np.meshgrid(w1_grid, w2_grid, indexing="ij")
    preds_grid = (
        W1[..., None] * x4_1[None, None, :]
        + W2[..., None] * x4_2[None, None, :]
        + b4
    )
    mse_grid = np.mean((preds_grid - y4_data[None, None, :]) ** 2, axis=-1)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "3D 회귀면 + 데이터점 + 표",
            "w₁–w₂–MSE 히트맵",
            "실제값 vs 예측값 (y=x 기준)",
            "손실곡면 (파라미터 공간)",
        ]
    )

    # 🔸 3D 회귀면 + 표 (데이터 공간)
    with tab1:
        # ✅ 여기서도 표 왼쪽, 그래프 오른쪽
        col_table, col_fig = st.columns([1, 2])

        with col_table:
            st.markdown("**데이터 표 (실제값 / 예측값 / 오차)**")
            st.dataframe(df_step4_view, use_container_width=True, height=480)

        with col_fig:
            grid_x1 = np.linspace(0, 5, 25)
            grid_x2 = np.linspace(0, 5, 25)
            GX1, GX2 = np.meshgrid(grid_x1, grid_x2)
            GY = w1 * GX1 + w2 * GX2 + b4

            fig_plane = go.Figure()
            fig_plane.add_scatter3d(
                x=x4_1,
                y=x4_2,
                z=y4_data,
                mode="markers",
                marker=dict(size=3, color="royalblue", opacity=0.8),
                name="실제 데이터",
            )

            # 회귀면: 단색(연한 회색) 평면
            fig_plane.add_surface(
                x=GX1,
                y=GX2,
                z=GY,
                surfacecolor=np.zeros_like(GY),
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                showscale=False,
                opacity=0.6,
                name="회귀면 (예측값)",
            )

            fig_plane.update_layout(
                title="Step 4: 다변수 데이터와 회귀면 (데이터 공간)",
                scene=dict(
                    xaxis_title="x1",
                    yaxis_title="x2",
                    zaxis_title="y",
                ),
                height=500,
            )
            st.plotly_chart(fig_plane, use_container_width=True)

            st.markdown(
                """
**그래프(데이터 공간) 읽는 법**

- 🔵 파란 점: 실제로 관측된 데이터 $(x_1, x_2, y)$  
- 회색 평면: 슬라이더에서 선택한 $(w₁, w₂, b)$로 계산한 예측값 $\\hat{y} = p(x_1, x_2)$  
- 평면이 점 구름을 **잘 가로지르면 → 예측이 실제와 비슷한 상태**,  
  평면이 점들과 멀리 떨어져 있으면 → **오차가 큰 상태**입니다.
"""
            )

    # 🔸 w1–w2–MSE 히트맵 (파라미터 공간 2D) → 여긴 그래프만
    with tab2:
        fig_heat = px.imshow(
            mse_grid,
            x=w2_grid,
            y=w1_grid,
            origin="lower",
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels={"x": "w₂", "y": "w₁", "color": "MSE"},
            title="w₁–w₂ 평면에서 MSE 히트맵 (b 고정)",
        )

        fig_heat.add_scatter(
            x=[w2],
            y=[w1],
            mode="markers",
            marker=dict(color="blue", size=10, symbol="x"),
            name="현재 선택한 (w₁, w₂)",
        )

        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown(
            """
**히트맵(파라미터 평면) 읽는 법**

- 축: 가로 = $w_2$, 세로 = $w_1$  
- 배경 색: **MSE(평균제곱오차)의 크기**  
  - 연한 노랑 → 상대적으로 **작은 오차**  
  - 진한 주황·빨강 → **큰 오차**  
- 🔵 파란 X: 지금 슬라이더로 선택한 **현재 (w₁, w₂)** 위치  
- 파란 X가 **노란 영역에 가까울수록 → 현재 파라미터가 "오차가 작은 좋은 조합"**입니다.
"""
        )

    # 🔸 실제 vs 예측 (y=x 기준선) → 그래프만
    with tab3:
        fig_pred = go.Figure()

        y_min = min(float(y4_data.min()), float(y4_hat.min()))
        y_max = max(float(y4_data.max()), float(y4_hat.max()))

        # 이상적 상황: y = x
        fig_pred.add_trace(
            go.Scatter(
                x=[y_min, y_max],
                y=[y_min, y_max],
                mode="lines",
                line=dict(dash="dash"),
                name="완벽한 예측선 (y = x)",
            )
        )

        # 실제 vs 예측 점
        fig_pred.add_trace(
            go.Scatter(
                x=y4_data,
                y=y4_hat,
                mode="markers",
                marker=dict(size=6, color="royalblue", opacity=0.8),
                name="실제 vs 예측",
            )
        )

        fig_pred.update_layout(
            title="실제값 vs 예측값 (점들이 y=x에 가까울수록 최적화 잘 된 것)",
            xaxis_title="실제 y",
            yaxis_title="예측 y_hat",
            height=450,
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown(
            """
**그래프(실제 vs 예측) 읽는 법**

- 점 하나 = 한 데이터의 (실제값 y, 예측값 $\\hat{y}$) 쌍  
- 회색 점선 **y = x**: "예측 = 실제"가 되는 이상적인 선  
- 점들이 y = x 선 위/근처에 몰릴수록  
  👉 우리가 선택한 $(w₁, w₂, b)$가 **데이터를 잘 맞추는 상태**입니다.
"""
        )

    # 🔸 손실곡면 3D (파라미터 공간 3D) + 표
    with tab4:
        # ✅ 표 왼쪽, 그래프 오른쪽
        col_table, col_fig = st.columns([1, 2])

        with col_fig:
            fig_loss = go.Figure()

            fig_loss.add_surface(
                x=w1_grid,
                y=w2_grid,
                z=mse_grid,
                colorscale="YlGnBu",  # 파스텔 톤
                opacity=0.9,
                name="손실곡면 L(w₁, w₂)",
            )

            fig_loss.add_scatter3d(
                x=[w1],
                y=[w2],
                z=[mse4],
                mode="markers",
                marker=dict(size=6, color="red"),
                name="현재 파라미터 (w₁, w₂)",
            )

            fig_loss.update_layout(
                title="손실함수 곡면 L(w₁, w₂) (파라미터 공간)",
                scene=dict(
                    xaxis_title="w₁",
                    yaxis_title="w₂",
                    zaxis_title="MSE",
                ),
                height=500,
            )

            st.plotly_chart(fig_loss, use_container_width=True)

        with col_table:
            st.markdown("**현재 선택한 파라미터 요약**")
            mse_table = pd.DataFrame(
                {
                    "파라미터": ["w₁", "w₂", "b", "MSE"],
                    "현재 값": [
                        round(float(w1), 3),
                        round(float(w2), 3),
                        round(float(b4), 3),
                        round(float(mse4), 4),
                    ],
                }
            )
            st.dataframe(mse_table, use_container_width=True, height=150)

            # 그리드 상의 최소 MSE 위치 계산
            min_idx = np.unravel_index(np.argmin(mse_grid), mse_grid.shape)
            best_w1 = float(w1_grid[min_idx[0]])
            best_w2 = float(w2_grid[min_idx[1]])
            best_mse = float(mse_grid[min_idx])

            st.markdown("**현재 MSE vs 최소 MSE (그리드 기준)**")
            compare_table = pd.DataFrame(
                {
                    "항목": ["현재 값", "그리드 상 최소값"],
                    "MSE": [round(mse4, 4), round(best_mse, 4)],
                    "w₁": [round(w1, 3), round(best_w1, 3)],
                    "w₂": [round(w2, 3), round(best_w2, 3)],
                }
            )
            st.dataframe(compare_table, use_container_width=True, height=180)

        st.markdown(
            """
**손실곡면(파라미터 공간 3D) 읽는 법**

- 축: 가로 = $w_1$, 세로 = $w_2$, 세로축(z) = MSE  
- 파스텔 톤 곡면: **각 파라미터 조합에서의 오차 크기**  
  - 낮은 지점(바닥) → 오차가 가장 작은 **최적의 파라미터**  
  - 높은 지점 → 오차가 큰 상태  
- 🔴 빨간 점: 지금 슬라이더에서 선택한 **현재 (w₁, w₂)**와 그때의 MSE 값  
- 오른쪽 표:  
  - 현재 $(w₁, w₂, b)$와 그때의 MSE  
  - 그리드 상에서 찾은 **최소 MSE 지점과 비교**  

👉 슬라이더를 움직이며, **빨간 점이 바닥 쪽으로 갈수록**  
   오른쪽 표에서 **현재 MSE가 최소 MSE에 가까워지는지** 같이 보세요.
"""
        )

    st.caption(
        "➡ 네 개의 탭은 서로 연결되어 있습니다. 같은 (w₁, w₂, b)가 데이터공간의 회귀면·오차·손실곡면을 동시에 바꿉니다."
    )

st.markdown("---")

# =========================
# Step 5. 최적화의 공통 구조
# =========================
st.header("🔹 Step 5. 공통 구조")

st.markdown(
    """
지금까지 네 가지 서로 다른 모양의 데이터를 보면서,  
**그 본질은 무엇인지 생각해봅시다.**

> **어떤 모양의 함수이든,  
> 오차를 줄이기 위해  
> 그 함수의 계수를 조절하는 과정.**

조금만 더 일반적으로 말하면:

1. **모델 선택**:  어떤 모양의 함수 $p(x; \\theta)$를 쓸지 정한다.  
2. **오차(손실) 정의**:  예측과 실제의 차이를 하나의 수 $L(\\theta)$로 정한다.  
3. **계수 조절**:  $L(\\theta)$가 줄어들도록 $\\theta$를 계속 바꿔본다.  
4. **멈추기**:  더 이상 눈에 띄게 줄지 않을 때, 그때의 $\\theta$를 "최적"이라고 부른다.
"""
)
st.caption( "조건이 두 개 이상 → 여러 요인이 동시에 작용하는 상황")

예시설명(
    "레시피 비율을 바꿔가며 최적의 맛을 찾는 과정 🍳",
    "온실 환경(온도·습도·광량)을 조절해 가장 빠르게 자라는 조건을 찾는 과정 🌿",
)

st.markdown("---")

st.success(
    "정리: 선형이든, 다항이든, 다변수든, 비선형이든 결국 **'파라미터를 조절해서 손실을 줄이는 최적화'**라는 같은 틀 안에 있다."
)
