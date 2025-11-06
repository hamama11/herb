import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="계수를 조절하는 최적화의 본질", layout="wide")

st.title("🎯 회귀했더니 000")

st.image("assets/회귀.png", use_container_width=600)

st.markdown("""
수학적으로 복잡해 보여도, **모든 회귀모델의 핵심 과정은 동일합니다.**  
> 📌 _오차가 최소가 되도록 모델의 **계수(parameter)** 를 조절하는 것_

---

### 🔹 Step 1. 선형 회귀 (Linear Regression)
- 모델:  $p(x) = a x + b$
- 조정 대상:  $a, b$
- 목표:  $\sum (y_i - (a x_i + b))^2$  최소화  
- 방법: 정규방정식, 경사하강법  

💡 직선의 기울기와 절편을 바꿔가며 오차를 줄이는 과정

---

### 🔹 Step 2. 다항 회귀 (Polynomial Regression)
- 모델:  $p(x) = a_0 + a_1x + a_2x^2 + ... + a_kx^k$
- 조정 대상:  $a_0, a_1, ..., a_k$
- 본질은 여전히 ‘계수 조절’
- 다항항이 늘어날수록 **계수가 많아지지만 원리는 동일**

💡 “곡선을 그리지만, 여전히 계수의 최적화 문제”

---

### 🔹 Step 3. 다변수 회귀 (Multiple Regression)
- 모델:  $p(x_1, x_2, ..., x_n) = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
- 조정 대상:  $w_1, w_2, ..., b$
- 오차함수:  $\text{SSE}(w) = \sum (y_i - p(x_i))^2$
- 최적화:  $\mathbf{w}_{new} = \mathbf{w}_{old} - \eta \nabla_\mathbf{w} \text{SSE}$

💡 여러 방향에서 동시에 오차를 줄이기 때문에 **기울기 벡터(gradient vector)** 를 사용

---

### 🔹 Step 4. 비선형 회귀 (Nonlinear Regression)
- 모델:  $p(x) = a e^{bx} + c \sin(dx)$
- 조정 대상:  $a, b, c, d$
- 오차함수는 **비선형**
- 해석적 해 불가능 → 수치적 방법 사용 (경사하강법, 뉴턴법, Adam 등)

💡 여전히 계수를 바꾸지만, **경로가 복잡하고 지역 최소에 빠질 위험 존재**

---

### 🔹 Step 5. 최적화의 일반화 가능한가?
모델은 다음 공식을 만족합니다:

$\displaystyle \text{Find } \theta = [a,b,c,\dots] \text{ that minimizes } L(\theta)$

- $\theta$: 모델의 모든 계수(parameter)
- $L(\theta)$: 오차함수(loss function)
- 목표:  **$L(\theta)$ 최소화 → 최적의 $\theta$**

---

✅ 결론:  
> “선형이든 비선형이든, 회귀든 분류든,  
> 결국 최적화는 **오차(손실)를 줄이기 위해 파라미터(계수)를 조절하는 과정**이다.”  
""")

st.info("💬 000에 들어갈 말은? ")

# 시각적 요약 다이어그램
st.markdown("---")
st.subheader("📈 계수 조절의 공통 구조 시각화")

fig = go.Figure()

# 노드 위치 정의
nodes = {
    "선형 회귀": (0, 0),
    "다항 회귀": (1, 0.3),
    "다변수 회귀": (2, 0),
    "비선형 회귀": (3, -0.3),
    "최적화 일반화": (4, 0)
}

# 연결선
edges = [
    ("선형 회귀", "다항 회귀"),
    ("다항 회귀", "다변수 회귀"),
    ("다변수 회귀", "비선형 회귀"),
    ("비선형 회귀", "최적화 일반화")
]

# 노드와 화살표 추가
for start, end in edges:
    x0, y0 = nodes[start]
    x1, y1 = nodes[end]
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.8,
        arrowcolor="royalblue"
    )

# 노드 점과 라벨
for name, (x, y) in nodes.items():
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers+text",
        marker=dict(size=20, color="lightblue", line=dict(width=2, color="royalblue")),
        text=[name],
        textposition="top center",
        hovertext=f"{name} 단계에서 조절되는 것은 계수(parameter)",
        hoverinfo="text"
    ))

fig.update_layout(
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    height=400,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

st.info("💬 **회귀와 최적화는 '계수를 조절하여 오차(손실)를 최소화하는 과정'.**")
