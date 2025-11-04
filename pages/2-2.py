import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="데이터 기반 최적화 실험", layout="wide")

st.title("📈 함수를 모를 때, 데이터로 최적점 추론하기")

st.markdown(
    """
    이 페이지는 다음 과정을 **하나의 실험 흐름**으로 보여줍니다.

    1. **데이터 점 찍기** (또는 예시 데이터 사용)  
    2. **형태 추론**: 대략 포물선 같네? → 다항식 모델 가정  
    3. **회귀(Regression)** 로 근사 함수 \\( \\hat{f}(x) \\) 만들기  
    4. **미분**으로 극값(최적점) 계산: \\( \\hat{f}'(x)=0 \\)  
    5. **검증**: 데이터와 근사곡선, 최적점 위치를 눈으로 확인  

    👉 수식 f(x)를 **직접 모르더라도**,  
    데이터 → 모델 → 미분 → 최적점 이라는 흐름으로 **간접 추론**을 경험하게 하는 페이지입니다.
    """
)

# ------------------------------------------------
# 1️⃣ 데이터 입력 / 예시 데이터
# ------------------------------------------------
st.header("1️⃣ 데이터 점 찍기: x–y 관계 관찰하기")

st.markdown(
    """
    - x: 입력 (예: 온실 온도, 광고비, 비료량 등)  
    - y: 결과 (예: 생장률, 매출, 수확량 등)  
    아래 표는 예시 데이터입니다. 마음대로 **값을 수정하거나 행을 추가/삭제**해 보세요.
    """
)

# 예시 데이터 (사용자가 수정 가능)
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

# NaN 제거
data_df = data_df.dropna()
if data_df.shape[0] < 3:
    st.error("최소 3개 이상의 데이터점이 필요합니다.")
    st.stop()

x = data_df["x"].to_numpy()
y = data_df["y"].to_numpy()

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

    같은 질문을 던지면서 **형태를 추측(model hypothesis)** 하는 단계입니다.
    """
)

# ------------------------------------------------
# 3️⃣ 근사 모델 설정 + 회귀로 모수 추정 (다항 회귀)
# ------------------------------------------------
st.header("3️⃣ 근사 모델: 다항식으로 \\( \\hat{f}(x) \\) 만들기")

degree = st.slider("다항식 차수 선택 (2~4)", min_value=2, max_value=4, value=2)

st.markdown(
    f"""
    - 실제 f(x)는 모르지만,  
      **“대략 포물선/부드러운 곡선 같네?”** 라는 전제 하에  
      **{degree}차 다항식**  
      \\[
      \\hat{{f}}(x) = a_{{{degree}}}x^{{{degree}}} + \\cdots + a_1x + a_0
      \\]
      로 근사해 봅니다.
    """
)

# numpy.polyfit으로 다항 회귀
coeffs = np.polyfit(x, y, deg=degree)  # 높은 차수부터 a_n, ..., a_0
p = np.poly1d(coeffs)  # 근사 다항함수
p_deriv = p.deriv()    # 도함수

# 곡선 그리기용 그리드
x_grid = np.linspace(float(x.min()), float(x.max()), 400)
y_pred = p(x_grid)

# 계수 표시용 LaTeX 문자열 생성
terms = []
pow_ = degree
for c in coeffs:
    c_round = np.round(c, 3)
    if pow_ > 1:
        terms.append(f"{c_round}x^{pow_}")
    elif pow_ == 1:
        terms.append(f"{c_round}x")
    else:
        terms.append(f"{c_round}")
    pow_ -= 1

poly_str = " + ".join(terms).replace("+ -", "- ")

st.latex(r"\hat{f}(x) = " + poly_str)

fig2, ax2 = plt.subplots()
ax2.scatter(x, y, alpha=0.5, label="데이터")
ax2.plot(x_grid, y_pred, label=f"{degree}차 다항 근사")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("데이터와 다항식 근사 곡선")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    """
    이 단계가 바로 **③ 데이터 피팅(모수 추정)** 단계입니다.  
    - 이제 우리는 실제 f(x)는 몰라도  
      **근사 함수 \\( \\hat{f}(x) \\)** 를 손에 쥐고 있습니다.
    """
)

# ------------------------------------------------
# 4️⃣ 극값 계산: 근사 모델 내부에서 최적점 추론 (미분)
# ------------------------------------------------
st.header("4️⃣ 극값 계산: \\( \\hat{f}'(x) = 0 \\) 에서 최적점 찾기")

# 도함수의 근(들) 계산
crit_points = np.roots(p_deriv)
real_crit = crit_points[np.isreal(crit_points)].real
# 데이터 범위 내에 있는 실근만 사용
real_crit_in_range = real_crit[(real_crit >= x.min()) & (real_crit <= x.max())]

if len(real_crit_in_range) == 0:
    st.warning("데이터 구간 안에 있는 도함수=0 실근이 없습니다. 차수나 데이터를 바꿔 보세요.")
else:
    xs_opt = real_crit_in_range
    ys_opt = p(xs_opt)
    # y가 최대인 점을 '최적점'으로 선택
    best_idx = np.argmax(ys_opt)
    x_opt = xs_opt[best_idx]
    y_opt = ys_opt[best_idx]

    fig3, ax3 = plt.subplots()
    ax3.scatter(x, y, alpha=0.3, label="데이터")
    ax3.plot(x_grid, y_pred, label=f"{degree}차 근사곡선")
    ax3.scatter(xs_opt, ys_opt, color="orange", s=60, label="극값 후보 (p'(x)=0)")
    ax3.scatter([x_opt], [y_opt], color="red", s=80, label="선택된 최적점")
    ax3.axvline(x_opt, linestyle="--", color="red")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("근사 모델 내에서의 극값(최적점) 탐색")
    ax3.legend()
    st.pyplot(fig3)

    st.markdown(
        f"""
        **도함수 \\( \\hat{{f}}'(x) = 0 \\) 에서 얻은 극값 후보들 중,  
        y가 가장 큰 지점을 '최적점'으로 잡으면:**

        - 추정 최적점 \\(x^*\\) ≈ **{x_opt:.3f}**  
        - 그 때의 \\( \\hat{{f}}(x^*) \\) ≈ **{y_opt:.3f}**

        여기서 중요한 포인트는:

        > 우리는 원래 f(x)를 전혀 모르지만,  
        > 데이터를 통해 만든 근사 함수 \\( \\hat{{f}}(x) \\) 안에서  
        > 완전히 **수학적 방식(미분)** 으로 최적점을 계산했다는 점입니다.
        """
    )

# ------------------------------------------------
# 5️⃣ 간단한 검증: 잔차와 R²
# ------------------------------------------------
st.header("5️⃣ 검증: 모델이 데이터를 얼마나 잘 설명하는가?")

# 예측값 (원 데이터 x에 대한)
y_hat = p(x)
resid = y - y_hat
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.metric("결정계수 R² (설명력)", f"{r2:.3f}")
    st.markdown(
        """
        - R²가 1에 가까울수록 **데이터 패턴을 잘 설명하는 모델**입니다.  
        - R²가 너무 낮다면,  
          - 모델 형태(차수)를 바꾸거나  
          - 데이터가 '증가→감소' 구조가 아닐 수 있습니다.
        """
    )

with col_res2:
    fig_res, ax_res = plt.subplots()
    ax_res.axhline(0, color="gray", linewidth=1)
    ax_res.scatter(x, resid)
    ax_res.set_xlabel("x")
    ax_res.set_ylabel("잔차 (y - ŷ)")
    ax_res.set_title("잔차(residual) 분포")
    st.pyplot(fig_res)

st.markdown(
    """
    이 단계가 **⑤ 검증 및 보정**에 해당합니다.  
    - 잔차가 특정 구간에서만 한쪽으로 치우치면,  
      > “모델이 그 부분에서는 패턴을 잘 못 잡고 있구나”  
      라고 해석할 수 있습니다.
    """
)

# ------------------------------------------------
# 6️⃣ 전체 흐름 재정리
# ------------------------------------------------
st.header("6️⃣ 전체 흐름 정리")

st.markdown(
    """
    우리가 방금 이 페이지에서 한 일을 한 줄로 요약하면:

    > **데이터 → 형태 추론 → 근사모델 → 미분적 탐색 → 검증**

    이었습니다.

    - 원래의 f(x)는 모르지만,  
      **패턴**을 보고 **가설 모델**을 세우고,  
      데이터를 이용해 모수를 추정한 뒤,  
      그 안에서 **수학적 최적화(미분)** 를 적용해 최적점을 찾았습니다.

    그대로 온실 데이터(온도–생장률),  
    혹은 기업 데이터(광고비–매출, 인력–생산량)에 가져가면  
    **“함수를 모를 때 최적점을 추론하는”** 표준적인 데이터 기반 최적화 흐름이 됩니다.
    """
)
