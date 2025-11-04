import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="데이터 기반 최적화 실험", layout="wide")

st.title("📈 함수를 모를 때, 데이터로 최적점 추론하기 (다항식·로그형·지수형 + 경사상승)")

st.markdown(
    """
    이 페이지는 **함수를 모를 때 데이터로 최적점을 추론하는 과정**을  
    여러 모델(다항식, 로그형, 지수형)과 **수치적 최적화(경사 상승)**까지 묶어 보여줍니다.

    1. **데이터 점 찍기** (또는 예시 데이터 사용)  
    2. **모델 가정**: 다항식 / 로그형 / 지수형 선택  
    3. 각 모델로 **회귀(Regression)** → 근사 함수 \\( \\hat{f}(x) \\) 만들기  
    4. 다항식은 **미분으로 극값(최적점)** 계산, 로그·지수형은 **경계값 비교**  
    5. **경사 상승(gradient ascent)** 으로 최적점에 가까워지는 자취 시각화  
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
# ------------------------------------------------
st.header("3️⃣ 근사 모델 설정: 다항식·로그형·지수형 가정하기")

st.markdown("**사용할 모델을 선택하세요. (복수 선택 가능)**")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    use_poly = st.checkbox("다항식 (Polynomial)", value=True)
with col_m2:
    use_log = st.checkbox("로그형 (Log)", value=False)
with col_m3:
    use_exp = st.checkbox("지수형 (Exponential)", value=False)

selected_degrees = []
if use_poly:
    st.markdown("**다항식 차수 선택 (여러 개 동시에 선택 가능)**")
    c2 = st.checkbox("2차", value=True, key="deg2")
    c3 = st.checkbox("3차", value=False, key="deg3")
    c4 = st.checkbox("4차", value=False, key="deg4")
    if c2:
        selected_degrees.append(2)
    if c3:
        selected_degrees.append(3)
    if c4:
        selected_degrees.append(4)

if not (use_poly or use_log or use_exp):
    st.error("적어도 하나의 모델(다항식, 로그형, 지수형)을 선택해야 합니다.")
    st.stop()

# ------------------------------------------------
# 공통 준비: R² 계산 함수
# ------------------------------------------------
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

# 다항식 모델 저장용 (경사상승에서 다시 쓰려고)
poly_models = {}  # deg -> poly1d

# ------------------------------------------------
# 4️⃣ 모델별 회귀 & 그래프 자취 그리기
# ------------------------------------------------
st.header("4️⃣ 데이터 피팅: 여러 모델의 자취를 한 번에 보기")

x_grid = np.linspace(float(x.min()), float(x.max()), 400)

models_info = []  # 각 모델의 정보 저장 (이름, 예측값, R², 최적점 등)

fig2, ax2 = plt.subplots()
ax2.scatter(x, y, alpha=0.4, label="데이터")

# --- 4-1. 다항식 모델들 ---
if use_poly and selected_degrees:
    for deg in selected_degrees:
        try:
            coeffs = np.polyfit(x, y, deg=deg)
            p = np.poly1d(coeffs)
            poly_models[deg] = p  # 나중에 경사상승에서 활용

            y_pred_grid = p(x_grid)
            y_hat = p(x)

            # R² 계산
            r2 = r2_score(y, y_hat)

            # 모델 이름
            name = f"다항식 {deg}차"

            # 자취(곡선) 그리기
            ax2.plot(x_grid, y_pred_grid, label=f"{name}")

            # 극값(최적점) 탐색: 도함수=0
            p_deriv = p.deriv()
            crit_points = np.roots(p_deriv)
            real_crit = crit_points[np.isreal(crit_points)].real
            real_crit_in_range = real_crit[(real_crit >= x.min()) & (real_crit <= x.max())]

            opt_x = None
            opt_y = None
            opt_note = ""

            if len(real_crit_in_range) > 0:
                xs_opt = real_crit_in_range
                ys_opt = p(xs_opt)
                best_idx = np.argmax(ys_opt)
                opt_x = xs_opt[best_idx]
                opt_y = ys_opt[best_idx]
                opt_note = f"내부 극대값 존재 (x≈{opt_x:.3f}, y≈{opt_y:.3f})"
            else:
                # 내부 극값 없으면 경계 비교
                y_min = p(x.min())
                y_max = p(x.max())
                if y_min >= y_max:
                    opt_x = x.min()
                    opt_y = y_min
                else:
                    opt_x = x.max()
                    opt_y = y_max
                opt_note = f"내부 극값 없음 → 구간 경계 최댓값 (x≈{opt_x:.3f}, y≈{opt_y:.3f})"

            models_info.append(
                {
                    "name": name,
                    "type": "polynomial",
                    "deg": deg,
                    "r2": r2,
                    "opt_x": opt_x,
                    "opt_y": opt_y,
                    "opt_note": opt_note,
                }
            )
        except np.linalg.LinAlgError:
            st.warning(f"{deg}차 다항식 회귀에 문제가 발생했습니다. 데이터 구성을 확인하세요.")

# --- 4-2. 로그형 모델 (y = a ln x + b) ---
if use_log:
    # x>0인 점만 사용
    mask_x = x > 0
    if np.sum(mask_x) < 2:
        st.warning("로그형 모델을 쓰려면 x>0인 데이터가 최소 2개 이상 필요합니다.")
    else:
        X_log = np.log(x[mask_x])
        Y_log = y[mask_x]
        # y = a ln x + b → ln x 를 설명변수로 하는 1차 회귀
        a, b = np.polyfit(X_log, Y_log, 1)

        def f_log(xx):
            return a * np.log(xx) + b

        y_pred_grid_log = f_log(x_grid[x_grid > 0])
        ax2.plot(
            x_grid[x_grid > 0],
            y_pred_grid_log,
            linestyle="--",
            label="로그형 (a ln x + b)",
        )

        y_hat_log = f_log(x[mask_x])
        r2_log = r2_score(y[mask_x], y_hat_log)

        # 로그형은 단순 a ln x + b 형태면 내부 극값 없음 (단조 증가/감소)
        # 최댓값은 구간 끝에서 결정
        y_min = f_log(x.min()) if x.min() > 0 else np.nan
        y_max = f_log(x.max())
        if np.isnan(y_min) or y_max >= y_min:
            opt_x_log = x.max()
            opt_y_log = y_max
        else:
            opt_x_log = x.min()
            opt_y_log = y_min

        models_info.append(
            {
                "name": "로그형",
                "type": "log",
                "deg": None,
                "r2": r2_log,
                "opt_x": opt_x_log,
                "opt_y": opt_y_log,
                "opt_note": "a ln x + b 형태는 단조이므로 최댓값은 구간 경계에서 발생",
            }
        )

# --- 4-3. 지수형 모델 (y = A e^{Bx}) ---
if use_exp:
    # y>0인 점만 사용 (로그 변환 위해)
    mask_y = y > 0
    if np.sum(mask_y) < 2:
        st.warning("지수형 모델을 쓰려면 y>0인 데이터가 최소 2개 이상 필요합니다.")
    else:
        X_exp = x[mask_y]
        Y_exp = y[mask_y]
        # ln y = ln A + B x → x를 설명변수로 하는 1차 회귀
        B, lnA = np.polyfit(X_exp, np.log(Y_exp), 1)
        A = np.exp(lnA)

        def f_exp(xx):
            return A * np.exp(B * xx)

        y_pred_grid_exp = f_exp(x_grid)
        ax2.plot(
            x_grid,
            y_pred_grid_exp,
            linestyle=":",
            label="지수형 (A e^{Bx})",
        )

        y_hat_exp = f_exp(x)
        r2_exp = r2_score(y, y_hat_exp)

        # 지수형도 단조 (B 부호에 따라) → 최댓값은 구간 경계
        y_min_exp = f_exp(x.min())
        y_max_exp = f_exp(x.max())
        if y_max_exp >= y_min_exp:
            opt_x_exp = x.max()
            opt_y_exp = y_max_exp
        else:
            opt_x_exp = x.min()
            opt_y_exp = y_min_exp

        models_info.append(
            {
                "name": "지수형",
                "type": "exp",
                "deg": None,
                "r2": r2_exp,
                "opt_x": opt_x_exp,
                "opt_y": opt_y_exp,
                "opt_note": "A e^{Bx} 형태는 단조이므로 최댓값은 구간 경계에서 발생",
            }
        )

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("데이터와 여러 모델의 자취")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    """
    - **다항식**은 차수별로 곡선이 겹쳐 그려져,  
      어느 모델이 데이터를 더 잘 따라가는지 **자취**를 비교할 수 있습니다.  
    - **로그형/지수형**은 “계속 증가/감소”하는 패턴을 표현할 때 적합하지만,  
      내부 극값(증가하다가 감소하는 꼭짓점)은 만들지 못합니다.
    """
)

# ------------------------------------------------
# 5️⃣ 모델별 R² & 최적점 요약
# ------------------------------------------------
st.header("5️⃣ 모델 비교: R²와 최적점 요약")

if models_info:
    summary_rows = []
    for m in models_info:
        summary_rows.append(
            {
                "모델": m["name"],
                "차수(다항식)": m["deg"] if m["deg"] is not None else "-",
                "R² (설명력)": np.round(m["r2"], 3),
                "추정 최적 x*": np.round(m["opt_x"], 3) if m["opt_x"] is not None else None,
                "추정 f(x*)": np.round(m["opt_y"], 3) if m["opt_y"] is not None else None,
                "비고": m["opt_note"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("계산된 모델 정보가 없습니다.")

st.markdown(
    """
    - **R²**는 각 모델이 데이터를 얼마나 잘 설명하는지 보여주는 지표입니다.  
    - 다항식 모델 중 R²가 높고, **내부 극값이 존재하는 모델**이  
      “증가하다가 감소하는” 현상을 설명하는 데 가장 자연스럽습니다.  
    - 로그형/지수형은 **최댓값이 구간 끝에서만 나타나기 때문에**,  
      “계속 증가 또는 감소”하는 상황에 더 어울리는 모델입니다.
    """
)

# ------------------------------------------------
# 6️⃣ 수치적 최적화: 경사 상승(gradient ascent) 자취
# ------------------------------------------------
st.header("6️⃣ 수치적 최적화: 경사 상승으로 최적점에 다가가기")

if not poly_models:
    st.info("경사 상승 실험은 다항식 모델이 있을 때만 가능합니다. (위에서 다항식을 하나 이상 선택하세요.)")
else:
    # 어떤 차수의 다항식으로 경사상승을 할지 선택
    deg_for_grad = st.selectbox(
        "경사 상승에 사용할 다항식 차수 선택",
        options=sorted(poly_models.keys()),
        format_func=lambda d: f"{d}차 다항식",
    )

    p_grad = poly_models[deg_for_grad]
    p_grad_deriv = p_grad.deriv()

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        x0 = st.slider(
            "초기값 x₀ (시작 위치)",
            float(x.min()),
            float(x.max()),
            float(np.median(x)),
            step=float((x.max() - x.min()) / 20),
        )
    with col_g2:
        lr = st.slider("학습률 (한 번에 이동하는 크기)", 0.001, 0.2, 0.05, 0.001)
    with col_g3:
        n_steps = st.slider("반복 횟수", 1, 50, 20)

    # gradient ascent: x_{k+1} = x_k + lr * f'(x_k)
    x_curr = x0
    xs_path = [x_curr]
    ys_path = [p_grad(x_curr)]

    for _ in range(n_steps):
        grad = p_grad_deriv(x_curr)
        x_curr = x_curr + lr * grad
        xs_path.append(x_curr)
        ys_path.append(p_grad(x_curr))

    fig_g, ax_g = plt.subplots()
    # 선택된 다항식 곡선
    x_grid_g = np.linspace(float(x.min()), float(x.max()), 400)
    ax_g.plot(x_grid_g, p_grad(x_grid_g), label=f"{deg_for_grad}차 다항식 근사")
    # 데이터
    ax_g.scatter(x, y, alpha=0.3, label="데이터")
    # 경사 상승 경로
    ax_g.plot(xs_path, ys_path, marker="o", linestyle="-", label="경사 상승 경로")
    ax_g.set_xlabel("x")
    ax_g.set_ylabel("y")
    ax_g.set_title("경사 상승으로 최적점 근처에 접근하는 자취")
    ax_g.legend()
    st.pyplot(fig_g)

    st.markdown(
        f"""
        - 초기값: \\(x_0 = {x0:.3f}\\)  
        - 반복 후 마지막 점: \\(x_{{\\text{{final}}}} ≈ {xs_path[-1]:.3f}\\),  
          \\( \\hat{{f}}(x_{{\\text{{final}}}}) ≈ {ys_path[-1]:.3f} \\)

        여기서 우리는 **전체 식을 다루는 대신**,  
        각 점에서의 **기울기(도함수 값)** 만 이용해 조금씩 더 높은 방향으로 올라갑니다.

        > 수학적 최적화:  
        > \\( \\hat{{f}}'(x) = 0 \\) 을 해석적으로 풀어서 극값을 “한 번에” 찾는다.  
        >
        > 수치적 최적화(경사 상승):  
        > \\( x_{{k+1}} = x_k + \\eta \\hat{{f}}'(x_k) \\) 을 반복하며  
        > 최적점 근처로 **점점 다가간다.**
        """
    )

# ------------------------------------------------
# 7️⃣ 전체 흐름 정리
# ------------------------------------------------
st.header("7️⃣ 전체 흐름 정리")

st.markdown(
    """
    지금까지 이 페이지에서 한 일을 한 줄로 정리하면:

    > **데이터 → 형태 추론 → (다항식/로그/지수) 모델 가정 → 회귀 →  
    > 다항식은 미분으로 내부 극값, 로그/지수는 경계값 비교 →  
    > 선택한 다항식에 대해 경사 상승으로 최적점 근사 → 모델 비교**

    입니다.

    수업에서는,
    - 온실 데이터(온도–생장률)를 넣고  
    - 다항식 2·3·4차, 로그형, 지수형을 동시에 켜고 끄면서  
    - **어떤 모델이 “증가→감소” 패턴을 가장 잘 설명하는지** 먼저 추론하게 한 뒤,  
    - 그중 하나의 다항식을 골라 **경사 상승 자취**를 보이면서

      > “미분으로 한 번에 찾는 극값” vs  
      > “기울기를 따라 조금씩 가는 수치적 최적화”

      의 차이를 경험하게 하면 좋습니다.
    """
)
