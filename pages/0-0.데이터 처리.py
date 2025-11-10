import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="데이터 분석 인포그래픽 - 온실 회귀", layout="wide")

# -----------------------------
# 0️⃣ 상단 히어로 섹션
# -----------------------------
col_hero1, col_hero2 = st.columns([2, 1])

with col_hero1:
    st.title("📊 데이터 분석 인포그래픽")
    st.subheader("온실 데이터를 통해 보는 회귀와 최적화의 큰 그림")
    st.markdown(
        """
        이 페이지는 **수업 전체의 지도(Map)** 역할을 하는 인포그래픽입니다.  
        - 위쪽: **데이터 분석 6단계 흐름**  
        - 중간: 데이터 분리와 회귀모델 스펙트럼  
        - 아래: 온실 데이터를 이용한 **회귀 예시 요약**  

        자세한 실습은 `회귀.py`, `함수모를때_최적화.py`에서 하고,  
        여기는 그 모든 내용을 **한 눈에 조망**하는 용도입니다.
        """
    )
with col_hero2:
    st.markdown("### 🌿 온실 시뮬레이션 데이터 개요")
    st.markdown(
        """
        - 온도(temperature)  
        - 습도(humidity)  
        - 생장률(growth)  

        세 변수를 가진 온실 데이터를 사용해  
        **회귀모델**과 **최적화**를 연결해서 이해해 봅니다.
        """
    )

st.markdown("---")

# -----------------------------
# 1️⃣ 온실 데이터 불러오기 / 샘플 생성
# -----------------------------
st.sidebar.header("데이터 설정")

uploaded_file = st.sidebar.file_uploader("온실 데이터 CSV 업로드 (선택)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ 업로드한 온실 데이터를 사용합니다.")
else:
    st.sidebar.info("업로드가 없으면 샘플 온실 데이터를 자동 생성합니다.")
    np.random.seed(0)
    n = 80
    temperature = np.linspace(15, 35, n)
    humidity = np.random.normal(65, 5, size=n)
    # 온도 26도, 습도 65 근처에서 생장률이 최대가 되도록 설계
    growth_clean = (
        -0.06 * (temperature - 26) ** 2
        + 0.012 * (humidity - 65)
        + 1.1
    )
    noise = np.random.normal(0, 0.06, size=n)
    growth = growth_clean + noise

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "growth": growth,
        }
    )

st.subheader("📁 온실 데이터 미리보기")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# 2️⃣ 데이터 분석 6단계 인포그래픽
# -----------------------------
st.markdown("## 1. 데이터 분석 6단계 흐름")

step_cols = st.columns(6)

steps = [
    ("1️⃣", "문제 정의", "무엇을 최적화할지 정의\n예: 생장률을 최대화하는 온도·습도 찾기"),
    ("2️⃣", "데이터 수집·탐색", "센서 데이터 수집, 분포·이상치 확인 (EDA)"),
    ("3️⃣", "전처리", "결측치/이상치 처리, 단위 통일, 스케일링 등"),
    ("4️⃣", "데이터 분리", "훈련/검증/테스트 세트로 나누기"),
    ("5️⃣", "모델링·최적화", "회귀모델 선택 + 손실 최소화로 계수 학습"),
    ("6️⃣", "평가·해석·보고", "오차·그래프·인사이트 정리 및 의사결정"),
]

for col, (no, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(f"### {no}")
        st.markdown(f"**{title}**")
        st.markdown(desc)

st.caption("➡ 이 6단계는 `회귀.py`, `함수모를때_최적화.py` 모든 실습의 **공통 뼈대**입니다.")

st.markdown("---")

# -----------------------------
# 3️⃣ 데이터 분리 인포그래픽 (슬라이더 제거 + 여러 방법 안내)
# -----------------------------
st.markdown("## 2. 데이터 분리: 훈련·검증·테스트를 나누는 여러 방법")

st.markdown(
    """
현실에서는 한 번에 모든 데이터를 모델 학습에 쓰지 않고,  
**역할에 따라 데이터를 나눠서** 사용합니다.
여기서는 예시로 **훈련 70% / 테스트 30%**를 가정해 봅니다.
"""
)

n_total = len(df)
train_ratio = 70
test_ratio = 30
n_train = int(n_total * train_ratio / 100)
n_test = n_total - n_train

col_donut1, col_donut2 = st.columns([1.2, 1])

with col_donut1:
    fig_split = px.pie(
        names=["Train", "Test"],
        values=[n_train, n_test],
        hole=0.6,
        color=["Train", "Test"],
        color_discrete_map={"Train": "#4F46E5", "Test": "#F97316"},
        title="예시: Train 70% / Test 30%"
    )
    fig_split.update_traces(textposition="inside", textinfo="label+percent")
    fig_split.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_split, use_container_width=True)

with col_donut2:
    st.markdown("### 📌 여러 가지 데이터 분리 방식")
    st.markdown(
        f"""
- 전체 샘플 수: **{n_total}개**  
- 예시 분리: 훈련 **{n_train}개 (70%)** / 테스트 **{n_test}개 (30%)**

일반적으로는 다음과 같은 방법들이 있습니다:

1. **단순 홀드아웃 (Hold-out)**  
   - 한 번만 `훈련 : 테스트 = 7:3` 또는 `8:2` 로 나눔  
   - 지금 예시처럼 가장 직관적이고 간단한 방법

2. **k-겹 교차 검증 (k-fold CV)**  
   - 데이터를 k조각으로 나누어,  
     그 중 하나는 테스트, 나머지는 훈련으로 번갈아 사용  
   - 작은 데이터에서 **안정적인 성능 추정**에 유리

3. **시계열 분리 (Time-based split)**  
   - 과거 데이터로 학습,  
     이후 시점의 데이터로 테스트  
   - 온실에서 **시간에 따른 변화 예측**을 할 때 적합

4. **층화 분리 (Stratified split)**  
   - 클래스 비율(예: 정상/불량)을 훈련·테스트에 비슷하게 맞춤  
   - 회귀가 아니라 분류 문제에서 자주 사용

`회귀.py`에서 보게 될 **MSE(평균제곱오차)** 도  
결국 이렇게 분리된 데이터(특히 테스트셋) 위에서 계산됩니다.
"""
    )

st.markdown("---")

# -----------------------------
# 4️⃣ 회귀 모델 스펙트럼
# -----------------------------
st.markdown("## 3. 회귀 모델 스펙트럼: 단순 → 복잡")

st.markdown(
    """
여러 회귀모델을 한 줄에 놓고 보면,  
**'얼마나 복잡한 모양을 그릴 수 있는가'**로 비교해 볼 수 있습니다.
"""
)

models = [
    "선형 회귀",
    "이차 다항 회귀",
    "고차 다항 회귀",
    "의사결정나무 회귀",
    "랜덤 포레스트",
    "XGBoost",
]
complexity = [1, 2, 3, 4, 5, 6]  # 개념적 복잡도 점수 (임의)

df_models = pd.DataFrame({"모델": models, "복잡도": complexity})

col_bar1, col_bar2 = st.columns([1.5, 1])

with col_bar1:
    fig_models = px.bar(
        df_models,
        x="복잡도",
        y="모델",
        orientation="h",
        text="복잡도",
        range_x=[0, 7],
        title="회귀모델 복잡도 비교 (개념적 수준)"
    )
    fig_models.update_traces(textposition="outside")
    fig_models.update_layout(
        height=400,
        xaxis_title="(단순)   ←   모델 복잡도   →   (복잡)"
    )
    st.plotly_chart(fig_models, use_container_width=True)

with col_bar2:
    st.markdown("### 어떻게 수업과 연결될까?")
    st.markdown(
        """
- **회귀.py**:  
  - 선형 회귀 → 다항 회귀 → 다변수 회귀 → 비선형 회귀  
  - 모두 **“계수를 조절해 오차를 줄이는 최적화”**라는 공통 구조

- **함수모를때_최적화.py**:  
  - 실제 함수는 몰라도,  
    온실 데이터를 이용해 **이차 다항식/다변수 회귀**로  
    온도·습도의 **최적 구간**을 추론

👉 모델이 복잡해질수록 더 다양한 패턴을 잡지만,  
그만큼 **과적합·해석 난이도**도 함께 올라갑니다.
"""
    )

st.markdown("---")

# -----------------------------
# 5️⃣ 온실 회귀 예시 (단변량/다변량) + 추정 모델식 (st.latex 사용)
# -----------------------------
st.markdown("## 4. 온실 회귀 실습 요약 (단변량 vs 다변량)")

col_uni, col_multi = st.columns(2)

# 🔹 단변량: 온도 → 생장률 (이차 다항 회귀)
with col_uni:
    st.markdown("### 🌡️ 온도 → 생장률 (이차 다항 회귀 요약)")
    x_temp = df["temperature"].values
    y_growth = df["growth"].values

    # 이차 다항 회귀
    coeffs = np.polyfit(x_temp, y_growth, deg=2)
    p2 = np.poly1d(coeffs)
    x_grid2 = np.linspace(x_temp.min(), x_temp.max(), 200)
    y_pred2 = p2(x_grid2)

    fig_uni = px.scatter(
        x=x_temp,
        y=y_growth,
        labels={"x": "Temperature (°C)", "y": "Growth"},
        title="온도 vs 생장률 + 2차 다항 회귀 곡선"
    )
    fig_uni.add_scatter(x=x_grid2, y=y_pred2, mode="lines", name="2차 다항 회귀")
    fig_uni.update_traces(marker=dict(size=6))
    fig_uni.update_layout(height=380)
    st.plotly_chart(fig_uni, use_container_width=True)

    a2, a1, a0 = coeffs
    y_hat_temp = p2(x_temp)
    mse_temp = np.mean((y_growth - y_hat_temp) ** 2)

    st.markdown("**추정된 모델식 (요약)**")
    # 🔑 st.latex로 수식 출력 (markdown + \\(...) 대신)
    st.latex(rf"\hat y = {a2:.3f} x^2 + {a1:.3f} x + {a0:.3f}")

    st.markdown(
        f"""
- 온도만을 사용하여 생장률을 설명하는 모델  
- MSE ≈ **{mse_temp:.4f}**

➡ `함수모를때_최적화.py`의 1변수–1결과 회귀와 연결됩니다.
"""
    )

# 🔹 다변량: 온도·습도 → 생장률 (다변수 선형 회귀)
with col_multi:
    st.markdown("### 🌡️+💧 온도·습도 → 생장률 (다변수 회귀 요약)")

    X = df[["temperature", "humidity"]].values
    y = df["growth"].values
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    b_hat = beta[0]
    w_temp, w_hum = beta[1:]

    y_hat_multi = X_design @ beta
    mse_multi = np.mean((y - y_hat_multi) ** 2)

    fig_multi = px.scatter(
        x=y,
        y=y_hat_multi,
        labels={"x": "실제 growth", "y": "예측 growth"},
        title="다변수 회귀: 실제값 vs 예측값"
    )
    fig_multi.add_scatter(
        x=[y.min(), y.max()],
        y=[y.min(), y.max()],
        mode="lines",
        name="이상적: y = ŷ",
        line=dict(dash="dash")
    )
    fig_multi.update_traces(marker=dict(size=6))
    fig_multi.update_layout(height=380)
    st.plotly_chart(fig_multi, use_container_width=True)

    st.markdown("**추정된 모델식 (선형)**")
    # 🔑 여기서도 st.latex 사용
    st.latex(
        rf"\hat y = {b_hat:.3f} + {w_temp:.3f}\,\text{{temp}} + {w_hum:.3f}\,\text{{humid}}"
    )

    st.markdown(
        f"""
- 온도와 습도를 **함께** 사용해 생장률을 설명하는 모델  
- MSE ≈ **{mse_multi:.4f}**

➡ `회귀.py`의 다변수 회귀,  
   `함수모를때_최적화.py`의 다변량 회귀 부분과 연결됩니다.
"""
    )

st.markdown("---")

# -----------------------------
# 6️⃣ 전체 요약
# -----------------------------
st.markdown("## 5. 한 장으로 정리하기")

st.markdown(
    """
- **위쪽**: 데이터 분석 6단계 (문제 정의 → EDA → 전처리 → 분리 → 모델링/최적화 → 평가/보고)  
- **중간**:  
  - 훈련/테스트 분리 인포그래픽  
  - 회귀모델 복잡도 스펙트럼  
- **아래**: 온실 데이터를 이용한  
  - 이차 다항 회귀 (온도 → 생장률)  
  - 다변수 회귀 (온도·습도 → 생장률)  

모든 내용을 한 줄로 요약하면 👇  

> **데이터에서 함수 모양(회귀모델)을 추정하고,  
> 그 함수 안에서 오차(손실)를 최소화하도록  
> 계수(파라미터)를 조절하는 과정 = 최적화**  
"""
)
