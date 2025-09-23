import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🎨 극좌표 활동지: 곡선의 길이와 넓이")

st.markdown("""
이 활동지는 **극좌표에서 면적과 곡선 길이**를 배우는 과정입니다.  
- **고2 Ver.** : 직관적 공식과 그림으로 이해  
- **고3 Ver.** : 좌표 변환과 미적분으로 공식 유도
""")

# ---------------------------
# 학생 입력 구간
# ---------------------------
st.header("1️⃣ 함수 입력하기")
func_choice = st.selectbox("r = f(θ)를 선택하세요", 
                           ["r = aθ", "r = 2 + θ", "r = 3sin(θ)", "직접 입력"])

if func_choice == "직접 입력":
    func_str = st.text_input("f(θ) = ", "θ + 1")
else:
    func_str = func_choice.split("=")[1].strip()

a = st.number_input("a 값 입력 (없으면 1로)", value=1.0)
theta_min = st.number_input("θ 최소값", value=0.0)
theta_max = st.number_input("θ 최대값", value=6.28)

st.write(f"👉 선택한 함수: r(θ) = {func_str}")

# ---------------------------
# 수학 함수 정의
# ---------------------------
theta = np.linspace(theta_min, theta_max, 500)

def f(theta):
    return eval(func_str, {"theta":theta, "np":np, "a":a})

r = f(theta)

# ---------------------------
# 그림 그리기
# ---------------------------
st.header("2️⃣ 곡선 그리기")

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, r, color="cyan")
ax.set_title("극좌표 곡선", fontsize=14)
st.pyplot(fig)

# ---------------------------
# 고2 활동: 공식 채우기
# ---------------------------
st.header("3️⃣ 고2 활동")
st.markdown("""
- 넓이 공식:  
\\[
A = \\tfrac{1}{2}\\int_{α}^{β} r^2 \\, dθ
\\]

- 길이 공식:  
\\[
L = \\int_{α}^{β} \\sqrt{r^2 + \\left(\\frac{dr}{dθ}\\right)^2} \\, dθ
\\]

👉 빈칸 채우기 활동:  
- 작은 조각의 넓이는 ( `?` )  
- 작은 조각의 길이는 ( `?` )
""")

# ---------------------------
# 고3 활동: 실제 계산
# ---------------------------
st.header("4️⃣ 고3 활동")

dr_dtheta = np.gradient(r, theta)
integrand_L = np.sqrt(r**2 + dr_dtheta**2)

area = 0.5 * np.trapz(r**2, theta)
length = np.trapz(integrand_L, theta)

st.latex(r"A = \tfrac{1}{2}\int r^2 d\theta \; \approx \; %.3f" % area)
st.latex(r"L = \int \sqrt{r^2 + \left(\tfrac{dr}{d\theta}\right)^2} d\theta \; \approx \; %.3f" % length)

st.markdown("""
👉 여기서 **고3 포인트**는:  
- 좌표 변환을 통해 공식을 유도할 수 있음  
- 수치적으로 적분해보면 실제 값도 확인할 수 있음
""")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🎨 극좌표 활동지: 곡선의 길이와 넓이")

st.markdown("""
이 활동지는 **극좌표에서 면적과 곡선 길이**를 배우는 과정입니다.  
- **고2 Ver.** : 직관적 공식과 그림으로 이해  
- **고3 Ver.** : 좌표 변환과 미적분으로 공식 유도
""")

# ---------------------------
# 학생 입력 구간
# ---------------------------
st.header("1️⃣ 함수 입력하기")
func_choice = st.selectbox("r = f(θ)를 선택하세요", 
                           ["r = aθ", "r = 2 + θ", "r = 3sin(θ)", "직접 입력"])

if func_choice == "직접 입력":
    func_str = st.text_input("f(θ) = ", "θ + 1")
else:
    func_str = func_choice.split("=")[1].strip()

a = st.number_input("a 값 입력 (없으면 1로)", value=1.0)
theta_min = st.number_input("θ 최소값", value=0.0)
theta_max = st.number_input("θ 최대값", value=6.28)

st.write(f"👉 선택한 함수: r(θ) = {func_str}")

# ---------------------------
# 수학 함수 정의
# ---------------------------
theta = np.linspace(theta_min, theta_max, 500)

def f(theta):
    return eval(func_str, {"theta":theta, "np":np, "a":a})

r = f(theta)

# ---------------------------
# 그림 그리기
# ---------------------------
st.header("2️⃣ 곡선 그리기")

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, r, color="cyan")
ax.set_title("극좌표 곡선", fontsize=14)
st.pyplot(fig)

# ---------------------------
# 고2 활동: 공식 채우기
# ---------------------------
st.header("3️⃣ 고2 활동")
st.markdown("""
- 넓이 공식:  
\\[
A = \\tfrac{1}{2}\\int_{α}^{β} r^2 \\, dθ
\\]

- 길이 공식:  
\\[
L = \\int_{α}^{β} \\sqrt{r^2 + \\left(\\frac{dr}{dθ}\\right)^2} \\, dθ
\\]

👉 빈칸 채우기 활동:  
- 작은 조각의 넓이는 ( `?` )  
- 작은 조각의 길이는 ( `?` )
""")

# ---------------------------
# 고3 활동: 실제 계산
# ---------------------------
st.header("4️⃣ 고3 활동")

dr_dtheta = np.gradient(r, theta)
integrand_L = np.sqrt(r**2 + dr_dtheta**2)

area = 0.5 * np.trapz(r**2, theta)
length = np.trapz(integrand_L, theta)

st.latex(r"A = \tfrac{1}{2}\int r^2 d\theta \; \approx \; %.3f" % area)
st.latex(r"L = \int \sqrt{r^2 + \left(\tfrac{dr}{d\theta}\right)^2} d\theta \; \approx \; %.3f" % length)

st.markdown("""
👉 여기서 **고3 포인트**는:  
- 좌표 변환을 통해 공식을 유도할 수 있음  
- 수치적으로 적분해보면 실제 값도 확인할 수 있음
""")
