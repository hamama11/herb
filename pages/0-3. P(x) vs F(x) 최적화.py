import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="함수를 모를 때 회귀로 추론하기", layout="wide")

# -----------------------------
# 0. 제목 + 안내
# -----------------------------
st.title("📈 함수를 모를 때, 회귀로 패턴 추론하기")

st.markdown(
    """
이전 페이지(📄 *회귀와 손실곡면*)에서는

> **모델식** $p(x; \\theta)$ 가 주어져 있을 때  
> 계수 $\\theta$를 바꾸며 손실 $L(\\theta)$를 최소화하는 과정을 봤습니다.

이번 페이지에서는 **실제 함수식을 모르는 상황**에서

1. 데이터를 눈으로 먼저 살펴보고  
2. **이차 다항식 회귀**로 1변수–1결과의 곡선 관계를 추정하고  
3. **다변수 선형 회귀**로 여러 입력이 하나의 결과에 미치는 영향을 동시에 추정합니다.

즉,  
> “함수를 모를 때, 회귀모델로 **가짜 함수**를 만들고  
>  그걸로 세상을 이해하려고 한다”  
는 관점을 체험해 봅니다.
"""
)

# -----------------------------
# 1️⃣ 데이터 불러오기 / 샘플 생성
# -----------------------------
st.sidebar.header("데이터 설정")

uploaded_file = st.sidebar.file_uploader("학습/점수 데이터 CSV 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 업로드한 데이터를 사용합니다.")
else:
    st.info("CSV를 업로드하지 않으면 **샘플 학습 데이터를 사용**합니다.")

    np.random.seed(0)
    n = 80

    # 샘플 데이터: 공부시간, 수면시간, 출석률 → 시험점수
    study_hours = np.random.uniform(0, 10, n)           # 하루 공부 시간
    sleep_hours = np.random.uniform(4, 9, n)            # 수면 시간
    attendance = np.random.uniform(70, 100, n)          # 출석률 (%)

    # "너무 안 하거나 너무 많이 해도 비효율" 같은 곡선을 만들기 위해
    # 공부시간에 대해 약간의 이차항 효과를 넣자.
    # (6시간 근처에서 가장 효율적이라는 설정)
    score_true = (
        -0.4 * (study_hours - 6) ** 2   # 공부시간에 대한 포물선 효과
        + 3.0 * sleep_hours
        + 0.3 * attendance
        + 40
    )
    noise = np.random.normal(0, 5, n)
    score = score_true + noise

    df = pd.DataFrame(
        {
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "attendance": attendance,
            "score": score,   # 시험 점수
        }
    )
    st.info("샘플 데이터 설명: study_hours, sleep_hours, attendance → score")

st.subheader("📊 데이터 미리보기")
st.dataframe(df.head())

# 숫자형 열만 추출
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
if len(numeric_cols) < 2:
    st.error("숫자형 열이 최소 2개 이상 필요합니다 (예: study_hours, score).")
    st.stop()

# -----------------------------
# 2️⃣ 1변수–1결과 설정 (이차 다항식 회귀용)
# -----------------------------
st.subheader("1️⃣ 1변수 → 1결과 관계 살펴보기 (이차 다항식 회귀)")

col1, col2 = st.columns(2)
with col1:
    x_col_1d = st.selectbox("입력 변수 (x) 선택", options=numeric_cols, index=0)
with col2:
    y_candidates_1d = [c for c in numeric_cols if c != x_col_1d]
    y_col_1d = st.selectbox("결과 변수 (y) 선택", options=y_candidates_1d, index=0)

x_1d = df[x_col_1d].values
y_1d = df[y_col_1d].values

# -----------------------------
# 3️⃣ 데이터 그대로 보기 (산점도)
# -----------------------------
st.markdown("#### (1) 산점도: 데이터 패턴만 먼저 보기")

fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(x_1d, y_1d, alpha=0.7)
ax_scatter.set_xlabel(x_col_1d)
ax_scatter.set_ylabel(y_col_1d)
ax_scatter.set_title(f"{x_col_1d} vs {y_col_1d}")
st.pyplot(fig_scatter)

st.markdown(
    f"""
- 아직 **함수식 f(x)** 는 모르는 상태입니다.  
- 단지,  
  > "{x_col_1d}가 늘어날수록 {y_col_1d}는 어떻게 변하나?"  
  를 눈으로 감각적으로만 보는 단계입니다.
"""
)

# -----------------------------
# 4️⃣ 이차 다항식 회귀
# -----------------------------
st.markdown("#### (2) 이차 다항식 회귀로 곡선 맞춰보기")

st.caption("모델:  $p(x) = a_2 x^2 + a_1 x + a_0$  (a₂, a₁, a₀를 데이터로부터 학습)")

# 이차 다항식 적합
coeffs_2 = np.polyfit(x_1d, y_1d, deg=2)  # [a2, a1, a0]
p2 = np.poly1d(coeffs_2)

x_grid = np.linspace(x_1d.min(), x_1d.max(), 400)
y_pred_2 = p2(x_grid)

fig_poly, ax_poly = plt.subplots()
ax_poly.scatter(x_1d, y_1d, alpha=0.5, label="데이터")
ax_poly.plot(x_grid, y_pred_2, color="orange", label="이차 다항식 회귀")
ax_poly.set_xlabel(x_col_1d)
ax_poly.set_ylabel(y_col_1d)
ax_poly.set_title("이차 다항식으로 근사한 곡선")
ax_poly.legend()
st.pyplot(fig_poly)

# 계수와 식 보여주기
a2, a1, a0 = coeffs_2
st.write(
    f"**추정된 모델식 (이차 다항식)**  \n"
    f"\\( \\hat y = {a2:.3f} x^2 + {a1:.3f} x + {a0:.3f} \\)"
)

y_hat_1d = p2(x_1d)
mse_1d = np.mean((y_1d - y_hat_1d) ** 2)
st.write(f"**이 모델의 MSE(평균제곱오차)**: {mse_1d:.3f}")

st.markdown(
    """
- 이제 우리는 원래 함수 f(x)는 몰라도,  
  데이터를 통해 **이차 다항식 모델** \\( p(x) \\) 을 얻었습니다.  

- 회귀.py에서 본 것처럼, 여기서도  
  > “계수(a₂, a₁, a₀)를 어떻게 잡아야  
  > 데이터와의 오차(MSE)가 가장 작아지는가?”  

  라는 최적화 문제가 **뒤에서 자동으로** 풀린 상태입니다  
  (여기서는 `np.polyfit`이 해줌).
"""
)

st.markdown("---")

# -----------------------------
# 5️⃣ 다변수 회귀 (여러 X → 하나의 y)
# -----------------------------
st.subheader("2️⃣ 다변수 회귀: 여러 입력이 하나의 결과에 미치는 영향")

st.markdown(
    """
이번에는 **여러 개의 입력 변수(X)** 를 동시에 사용해서  
하나의 결과(y)를 예측하는 **다변수 선형 회귀**를 살펴봅니다.

모델:  \n
\\[
\\hat y = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b
\\]
"""
)

# 대상 y, X 선택
col_my1, col_my2 = st.columns(2)
with col_my1:
    target_col = st.selectbox("결과 변수 (y) 선택", options=numeric_cols, index=len(numeric_cols) - 1)
with col_my2:
    feature_candidates = [c for c in numeric_cols if c != target_col]
    feature_cols = st.multiselect(
        "입력 변수들 (X) 선택 (2개 이상 추천)",
        options=feature_candidates,
        default=feature_candidates[:2],
    )

if len(feature_cols) == 0:
    st.warning("하나 이상의 입력 변수를 선택해 주세요.")
else:
    X = df[feature_cols].values  # (N, d)
    y_multi = df[target_col].values  # (N, )

    # 설계 행렬 (intercept 포함)
    X_design = np.column_stack([np.ones(X.shape[0]), X])  # (N, d+1)

    # 최소제곱 해 구하기: beta = (X^T X)^(-1) X^T y
    beta, *_ = np.linalg.lstsq(X_design, y_multi, rcond=None)
    b_hat = beta[0]
    w_hat = beta[1:]

    y_hat_multi = X_design @ beta
    mse_multi = np.mean((y_multi - y_hat_multi) ** 2)

    # 계수 표
    st.markdown("#### (1) 학습된 계수들")

    coef_table = pd.DataFrame({
        "항목": ["절편 b"] + [f"w ({col})" for col in feature_cols],
        "값": [b_hat] + list(w_hat),
    })
    coef_table["값"] = coef_table["값"].round(4)
    st.dataframe(coef_table, use_container_width=True, height=200)

    # 실제 vs 예측
    st.markdown("#### (2) 실제값 vs 예측값")

    fig_mv, ax_mv = plt.subplots()
    ax_mv.scatter(y_multi, y_hat_multi, alpha=0.7)
    min_y = min(y_multi.min(), y_hat_multi.min())
    max_y = max(y_multi.max(), y_hat_multi.max())
    ax_mv.plot([min_y, max_y], [min_y, max_y], "k--", label="이상적: y = ŷ")
    ax_mv.set_xlabel("실제값 (y)")
    ax_mv.set_ylabel("예측값 (ŷ)")
    ax_mv.set_title("다변수 회귀: 실제 vs 예측")
    ax_mv.legend()
    st.pyplot(fig_mv)

    st.write(f"**다변수 회귀 MSE(평균제곱오차)**: {mse_multi:.3f}")

    st.markdown(
        f"""
- 점들이 **y = ŷ** 선 근처에 몰릴수록,  
  선택한 입력 변수 {feature_cols}만으로도  
  {target_col}을 꽤 잘 설명할 수 있다는 뜻입니다.

- 여기서도 회귀.py에서 본 것처럼,  
  > "w₁, w₂, ..., b를 어떻게 정해야 오차가 최소인가?"  
  라는 **최적화 문제**를  
  `np.linalg.lstsq`가 자동으로 풀어준 결과입니다.
"""
    )

st.markdown("---")

# -----------------------------
# 6️⃣ 전체 정리
# -----------------------------
st.subheader("3️⃣ 정리: 함수를 모를 때 회귀로 하는 일")

st.markdown(
    """
1. **데이터 관찰**  
   - 먼저 산점도로 패턴을 본다. (증가, 감소, 포물선, 한계효과 등)

2. **모델 형태 가정 (회귀)**  
   - 1변수라면 이차 다항식 \\( a_2 x^2 + a_1 x + a_0 \\)  
   - 여러 변수라면 선형 결합 \\( w_1 x_1 + \\dots + w_n x_n + b \\)  
   같은 **함수 모양을 먼저 정한다.**

3. **계수 학습 = 최적화**  
   - 데이터를 보고,  
     > 어떤 계수(a₂, a₁, a₀), (w₁, ..., b)가  
     > 오차(MSE)를 가장 작게 만드는가?  
   를 푸는 것이 곧 **최적화**입니다.  
   (여기서는 `polyfit`, `lstsq`가 뒤에서 해결)

👉 함수 f(x)를 모를 때도,  
결국은 **회귀모델로 '가짜 함수'를 만들고  
그 안에서 최적화를 수행하는 것**이  
실제 데이터 분석·머신러닝에서 매우 흔한 패턴입니다.
"""
)
