import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="온실 데이터로 배우는 최적화", layout="wide")

st.title("🌱 온실 데이터로 배우는 '함수를 모를 때' 최적화")
st.markdown(
    """
    온실에서 수집한 데이터를 가지고 **함수식을 모르는 상태**에서  
    '어떤 조건(예: 온도)이 가장 좋을까?'를 추론하는 과정을 살펴봅니다.
    
    1. 데이터 그대로 보기 (패턴 관찰)  
    2. 다항식으로 근사해서 '가짜 함수' 만들기  
    3. 그 함수로 **수학적 최적화(미분)**, **수치적 최적화(경사 상승)** 비교  
    """
)

# -----------------------------
# 1️⃣ 데이터 불러오기 / 샘플 생성
# -----------------------------
st.sidebar.header("데이터 설정")

uploaded_file = st.sidebar.file_uploader("온실 데이터 CSV 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("업로드한 온실 데이터를 사용합니다.")
else:
    st.info("CSV를 업로드하지 않으면 샘플 온실 데이터를 사용합니다.")
    np.random.seed(0)
    # 샘플: 온도 15~35도, 최적 온도 26도 근처에서 생장률 최대
    temperature = np.linspace(15, 35, 60)
    true_opt = 26
    growth_clean = -0.07 * (temperature - true_opt) ** 2 + 1.2
    noise = np.random.normal(0, 0.05, size=temperature.size)
    growth = growth_clean + noise
    humidity = np.random.normal(65, 4, size=temperature.size)

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "growth": growth,   # 예: 생장률, 잎 면적 증가량 등
        }
    )

st.subheader("온실 데이터 미리보기")
st.dataframe(df.head())

numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
if len(numeric_cols) < 2:
    st.error("숫자형 열이 최소 2개 이상 필요합니다 (예: temperature, growth).")
    st.stop()

# -----------------------------
# 2️⃣ x, y 변수 선택
# -----------------------------
st.subheader("분석할 변수 선택")

col1, col2 = st.columns(2)
with col1:
    x_col = st.selectbox("입력 변수 (x) 예: 온도, 습도", options=numeric_cols, index=0)
with col2:
    y_candidates = [c for c in numeric_cols if c != x_col]
    y_col = st.selectbox("결과 변수 (y) 예: 생장률, 수확량", options=y_candidates, index=0)

x = df[x_col].values
y = df[y_col].values

# -----------------------------
# 3️⃣ 데이터 그대로 보기 (패턴 관찰)
# -----------------------------
st.subheader("1️⃣ 데이터 그대로 보기: 증가→감소 패턴이 보이나?")
fig1, ax1 = plt.subplots()
ax1.scatter(x, y)
ax1.set_xlabel(x_col)
ax1.set_ylabel(y_col)
ax1.set_title(f"{x_col} vs {y_col}")
st.pyplot(fig1)

st.markdown(
    f"""
    - 이 단계는 **'함수를 모를 때' 1단계: 패턴 관찰** 단계입니다.  
    - 학생들에게  
      > "{x_col}가 커질수록 {y_col}는 계속 증가하나요? 어느 지점 이후에는 줄어드는 것처럼 보이나요?"  
      라는 질문을 던질 수 있어요.
    """
)

# -----------------------------
# 4️⃣ 다항 회귀로 '가짜 함수' 만들기
# -----------------------------
st.subheader("2️⃣ 다항식으로 근사: 모르는 함수를 '가짜 함수'로 추론하기")

degree = st.slider("다항식 차수 선택 (2~4)", min_value=2, max_value=4, value=2)

# np.polyfit으로 다항 회귀
coeffs = np.polyfit(x, y, deg=degree)  # 높은 차수부터 계수
p = np.poly1d(coeffs)                   # 근사 함수 p(x)
p_deriv = p.deriv()                     # 도함수 p'(x)

# 곡선 그리기
x_grid = np.linspace(x.min(), x.max(), 400)
y_pred = p(x_grid)

fig2, ax2 = plt.subplots()
ax2.scatter(x, y, alpha=0.5, label="데이터")
ax2.plot(x_grid, y_pred, label=f"다항 근사 (차수={degree})")
ax2.set_xlabel(x_col)
ax2.set_ylabel(y_col)
ax2.set_title("온실 데이터 다항 회귀 근사")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    f"""
    - 이제 우리는 **실제 함수 f(x)** 는 모르지만,  
      데이터를 이용해 **근사 함수** \\( \\hat f(x) = p(x) \\) 를 만들었습니다.  
    - 이건 실제 기업에서도 많이 쓰는 방식입니다.  
      (실험 데이터 → 회귀 모델 → 그 모델 안에서 최적화)
    """
)

# -----------------------------
# 5️⃣ (A) 수학적 최적화: p'(x) = 0 단계
# -----------------------------
st.subheader("3️⃣ 수학적 최적화: 근사 함수의 도함수 p′(x)=0으로 최적점 찾기")

# 도함수의 근(들) 찾기
crit_points = np.roots(p_deriv)  # 복소근 포함
real_crit = crit_points[np.isreal(crit_points)].real
# 데이터 범위 안에 있는 점만
real_crit_in_range = real_crit[(real_crit >= x.min()) & (real_crit <= x.max())]

if len(real_crit_in_range) > 0:
    # y 값이 최대인 지점을 "최적점" 후보로 선택
    xs_opt = real_crit_in_range
    ys_opt = p(xs_opt)
    best_idx = np.argmax(ys_opt)
    x_opt_math = xs_opt[best_idx]
    y_opt_math = ys_opt[best_idx]

    fig3, ax3 = plt.subplots()
    ax3.plot(x_grid, y_pred, label="다항 근사")
    ax3.scatter(x, y, alpha=0.3, label="데이터")
    ax3.scatter([x_opt_math], [y_opt_math], s=80, marker="o", label="수학적 최적점")
    ax3.axvline(x_opt_math, linestyle="--")
    ax3.set_xlabel(x_col)
    ax3.set_ylabel(y_col)
    ax3.set_title("도함수=0 지점에서 찾은 최적점")
    ax3.legend()
    st.pyplot(fig3)

    st.write(
        f"**도함수 p′(x)=0에서 나온 최적점 (근사):**  \n"
        f"- {x_col} ≈ **{x_opt_math:.2f}** 일 때  {y_col} ≈ **{y_opt_math:.3f}**"
    )
else:
    st.warning("이 다항식의 도함수=0 실근이 데이터 구간 안에 없습니다. (차수/데이터를 바꿔보세요.)")

st.markdown(
    """
    - 여기까지가 **'모델을 세운 뒤 미분으로 극값을 찾는' 수학적 최적화**입니다.  
    - 원래 함수 f(x)는 몰라도, **근사함수 p(x)** 가 생기면  
      > p′(x)=0 → 극값 후보 → 그 중 최대인 지점 = 최적점  
      으로 논리를 전개할 수 있습니다.
    """
)

# -----------------------------
# 6️⃣ (B) 수치적 최적화: 경사 상승(gradient ascent)
# -----------------------------
st.subheader("4️⃣ 수치적 최적화: 경사(기울기)를 따라 한 걸음씩 올라가기")

col_g1, col_g2, col_g3 = st.columns(3)
with col_g1:
    x0 = st.slider(
        "초기값 x₀ (시작하는 온도)",
        float(x.min()),
        float(x.max()),
        float(np.median(x)),
        step=float((x.max() - x.min()) / 20),
    )
with col_g2:
    lr = st.slider("학습률 (한 번에 이동하는 크기)", 0.001, 0.2, 0.05, 0.001)
with col_g3:
    n_steps = st.slider("반복 횟수", 1, 50, 20)

# gradient ascent: x_{k+1} = x_k + lr * p'(x_k)
x_curr = x0
xs_path = [x_curr]
ys_path = [p(x_curr)]

for _ in range(n_steps):
    grad = p_deriv(x_curr)
    x_curr = x_curr + lr * grad  # 최대값을 찾는 '경사 상승'
    xs_path.append(x_curr)
    ys_path.append(p(x_curr))

fig4, ax4 = plt.subplots()
ax4.plot(x_grid, y_pred, label="다항 근사")
ax4.scatter(x, y, alpha=0.3, label="데이터")
ax4.plot(xs_path, ys_path, marker="o", linestyle="-", label="경사 상승 경로")
ax4.set_xlabel(x_col)
ax4.set_ylabel(y_col)
ax4.set_title("경사 상승으로 근사 최적점에 접근하는 과정")
ax4.legend()
st.pyplot(fig4)

st.write(
    f"**경사 상승 후 마지막 점 (근사):**  \n"
    f"- {x_col} ≈ **{xs_path[-1]:.2f}**,  {y_col} ≈ **{ys_path[-1]:.3f}**"
)

st.markdown(
    """
    - 이 과정이 바로 **'수치적 최적화(gradient ascent)'**입니다.  
    - 수식 전체를 알지 못해도,  
      > '지금 위치에서의 기울기'를 이용해  
      > 조금씩 더 좋은 방향으로 움직이는 전략  
      을 반복하다 보면 최적점 근처에 도달합니다.  
    - 실제 기업의 AI·머신러닝에서 쓰는 경사하강/상승법도 이 원리입니다.
    """
)

# -----------------------------
# 7️⃣ 마무리 비교
# -----------------------------
st.subheader("5️⃣ 정리: 함수 모를 때 최적점 찾기 3단계 요약")

st.markdown(
    f"""
    1. **패턴 관찰 (EDA)**  
       - 온실 데이터에서 {x_col}–{y_col} 그래프를 보고  
         “어디까지는 증가, 그 이후에는 감소”인지 눈으로 확인.
    2. **근사 모델 만들기 (회귀)**  
       - 실제 함수 f(x)는 모르지만, 다항식 p(x)를 학습해서  
         '가짜 함수'를 만든 뒤 그 안에서 최적점 탐색.
    3. **최적화 방법 적용**  
       - 수학적: p′(x)=0인 지점들 중에서 {y_col}를 최대로 만드는 x 선택  
       - 수치적: 기울기 p′(x)를 이용해 x를 조금씩 이동시키며 최댓값 근처로 수렴  

    👉 이렇게 하면 **온실 온도/습도에 대한 최적 조건**을  
       함수식을 모르는 상황에서도 데이터만으로 추론할 수 있습니다.
    """
)
