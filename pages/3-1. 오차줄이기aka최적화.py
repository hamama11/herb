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

with st.expander("👉 w₁, w₂, b를 조절하면서 회귀면과 오차를 시각화", expanded=False):
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

    # 데이터프레임 구성
    df_multi = {
        "x1": x1,
        "x2": x2,
        "y": y_multi,
        "y_hat": y_hat_multi,
        "오차": y_multi - y_hat_multi,
    }

    tab1, tab2, tab3 = st.tabs([
        "3D 회귀면 + 데이터점",
        "w₁–w₂–MSE 히트맵",
        "실제값 vs 예측값"
    ])

    # 🔸 3D 회귀면 시각화
    with tab1:
        # 평면용 grid
        grid_x1 = np.linspace(0, 5, 25)
        grid_x2 = np.linspace(0, 5, 25)
        GX1, GX2 = np.meshgrid(grid_x1, grid_x2)
        GY = w1 * GX1 + w2 * GX2 + b_mv

        fig_plane = go.Figure()

        # 실제 데이터 점
        fig_plane.add_scatter3d(
            x=x1,
            y=x2,
            z=y_multi,
            mode="markers",
            marker=dict(size=3, color="royalblue", opacity=0.8),
            name="데이터"
        )

        # 회귀면
        fig_plane.add_surface(
            x=GX1,
            y=GX2,
            z=GY,
            colorscale="RdBu",
            opacity=0.6,
            name="회귀면"
        )

        fig_plane.update_layout(
            title="회귀면과 데이터의 위치 관계",
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title="y"
            ),
            height=500,
        )
        st.plotly_chart(fig_plane, use_container_width=True)

    # 🔸 w1-w2 히트맵 (b 고정)
    with tab2:
        w1_grid = np.linspace(0.0, 3.0, 40)
        w2_grid = np.linspace(0.0, 2.0, 40)
        W1, W2 = np.meshgrid(w1_grid, w2_grid, indexing="ij")
        preds = W1[..., None] * x1[None, None, :] + W2[..., None, :] * x2[None, None, :] + b_mv
        mse_grid = np.mean((preds - y_multi[None, None, :]) ** 2, axis=-1)

        fig_heat = px.imshow(
            mse_grid,
            x=w2_grid,
            y=w1_grid,
            origin="lower",
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels={"x": "w₂", "y": "w₁", "color": "MSE"},
            title="w₁–w₂ 평면에서 MSE 히트맵 (b 고정)"
        )
        fig_heat.add_scatter(x=[w2], y=[w1], mode="markers",
                             marker=dict(color="blue", size=8),
                             name="현재 (w₁, w₂)")
        st.plotly_chart(fig_heat, use_container_width=True)

    # 🔸 실제 vs 예측
    with tab3:
        fig_pred = px.scatter(
            x=y_multi, y=y_hat_multi,
            labels={"x": "실제 y", "y": "예측 y_hat"},
            title="실제값 vs 예측값"
        )
        fig_pred.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_pred, use_container_width=True)

    st.caption("➡ 여러 변수와 계수가 있어도, 여전히 하는 일은 **w₁, w₂, b를 잘 골라 오차를 줄이는 것**입니다.")

st.markdown("---")

# =========================
# Step 5. 수식 대신 쉬운 말로 정리
# =========================
st.header("🔹 Step 5. 최적화의 공통 구조")

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

st.markdown("---")

st.success("정리: 선형이든, 다항이든, 다변수든, 비선형이든 결국 **'계수를 조절해서 오차를 줄이는 최적화'**라는 같은 틀 안에 있다.")
