# -*- coding: utf-8 -*-

+ import io
+ import numpy as np
+ import pandas as pd
+ import altair as alt
+ import streamlit as st
+
+ def show():
+     st.title("🌿 온실 관리 페이지")
+     st.write("여기는 온실 상태를 관리하는 페이지입니다.")
"""
🌱 Greenhouse Math: Streamlit Web App
- Upload `greenhouse.csv` or use the sample template
- Explore time series & relationships (Altair)
- Fit simple models (linear & optional quadratic via numpy.polyfit)
- Predict growth with interactive sliders

Expected columns (Korean headers by default):
- 날짜, 평균온도, 습도, 광량, 잎길이, 식물
"""

st.set_page_config(page_title="🌿 온실 속 수학자", layout="wide")
st.title("🌿 온실 속 수학자: 데이터·수학·코딩")

# ------------------------------
# Helpers
# ------------------------------
DEFAULT_COLUMNS = ["날짜", "평균온도", "습도", "광량", "잎길이", "식물"]

SAMPLE_CSV = """날짜,평균온도,습도,광량,잎길이,식물
2025-10-25,23.1,62,18000,4.2,바질
2025-10-26,24.8,58,22000,4.9,바질
2025-10-27,26.0,55,25000,5.6,바질
2025-10-28,27.1,52,26000,6.2,바질
2025-10-29,26.4,54,24000,6.6,바질
2025-10-30,24.2,60,21000,6.8,바질
2025-10-31,22.9,65,17000,6.9,바질
2025-10-25,22.4,64,16000,3.8,민트
2025-10-26,23.6,61,19000,4.3,민트
2025-10-27,25.1,57,23000,4.9,민트
2025-10-28,26.3,54,25500,5.5,민트
2025-10-29,25.7,56,23500,5.8,민트
2025-10-30,23.9,62,20000,6.0,민트
2025-10-31,22.7,66,16500,6.1,민트
"""

def download_button_for_template():
    st.download_button(
        label="📥 샘플 CSV 내려받기 (greenhouse.csv)",
        data=SAMPLE_CSV.encode("utf-8"),
        file_name="greenhouse.csv",
        mime="text/csv",
        help="예상 컬럼: 날짜, 평균온도, 습도, 광량, 잎길이, 식물"
    )

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 표준 컬럼명 보정 시도 (영/한 혼합 대비)
    rename_map = {}
    for col in df.columns:
        cc = col.strip().lower()
        if cc in ["date", "날짜"]: rename_map[col] = "날짜"
        elif cc in ["temp", "temperature", "평균온도", "온도"]: rename_map[col] = "평균온도"
        elif cc in ["humidity", "습도"]: rename_map[col] = "습도"
        elif cc in ["light", "lux", "광량", "조도"]: rename_map[col] = "광량"
        elif cc in ["leaf", "leaf_length", "잎길이", "잎 길이"]: rename_map[col] = "잎길이"
        elif cc in ["plant", "species", "식물", "식물명"]: rename_map[col] = "식물"
    df = df.rename(columns=rename_map)

    # 날짜 처리
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 숫자 처리
    for c in ["평균온도", "습도", "광량", "잎길이"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 식물 결측 대체
    if "식물" in df.columns:
        df["식물"] = df["식물"].fillna("미상")

    # 필수 컬럼만 유지
    cols = [c for c in DEFAULT_COLUMNS if c in df.columns]
    df = df[cols]

    # 결측 제거
    df = df.dropna(subset=[c for c in ["날짜", "평균온도", "습도", "광량", "잎길이"] if c in df.columns])

    return df

# ------------------------------
# Sidebar: Data input
# ------------------------------
st.sidebar.header("데이터 업로드")
uploaded = st.sidebar.file_uploader("greenhouse.csv 선택", type=["csv"]) 

c1, c2 = st.columns([2,1])
with c1:
    if uploaded is not None:
        try:
            raw = load_data(uploaded)
        except Exception as e:
            st.error(f"CSV를 읽는 중 오류: {e}")
            st.stop()
    else:
        st.info("CSV를 업로드하지 않으면 샘플 데이터를 사용합니다.")
        raw = pd.read_csv(io.StringIO(SAMPLE_CSV))
        download_button_for_template()

    df = clean_data(raw)
    if df.empty or not set(["날짜","평균온도","습도","광량","잎길이"]).issubset(df.columns):
        st.error("데이터에 필요한 컬럼이 부족합니다. 필요한 컬럼: 날짜, 평균온도, 습도, 광량, 잎길이 (식물은 선택)")
        st.stop()

    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.sort_values("날짜"))

with c2:
    st.subheader("필터")
    plants = ["(전체)"] + sorted(df["식물"].dropna().unique().tolist()) if "식물" in df.columns else ["(전체)"]
    sel_plant = st.selectbox("식물 선택", plants)

# Filter by plant
if sel_plant != "(전체)" and "식물" in df.columns:
    vdf = df[df["식물"] == sel_plant].copy()
else:
    vdf = df.copy()

# ------------------------------
# Charts
# ------------------------------
left, right = st.columns(2)
with left:
    st.subheader("📈 날짜별 잎 길이 변화")
    chart1 = (
        alt.Chart(vdf)
        .mark_line(point=True)
        .encode(
            x=alt.X("날짜:T", title="날짜"),
            y=alt.Y("잎길이:Q", title="잎 길이(cm)"),
            color=alt.condition(alt.datum.식물 != None, alt.Color("식물:N"), alt.value("#999")),
            tooltip=["날짜:T","평균온도:Q","습도:Q","광량:Q","잎길이:Q","식물:N"]
        )
        .properties(height=320)
    )
    st.altair_chart(chart1, use_container_width=True)

with right:
    st.subheader("🌡️ 온도 vs 잎길이")
    chart2 = (
        alt.Chart(vdf)
        .mark_circle()
        .encode(
            x=alt.X("평균온도:Q", title="평균 온도(℃)"),
            y=alt.Y("잎길이:Q", title="잎 길이(cm)"),
            color=alt.Color("식물:N", title="식물") if "식물" in vdf.columns else alt.value("#1f77b4"),
            tooltip=["날짜:T","평균온도:Q","습도:Q","광량:Q","잎길이:Q","식물:N"]
        )
        .properties(height=320)
    )
    st.altair_chart(chart2, use_container_width=True)

# ------------------------------
# Modeling
# ------------------------------
st.subheader("🧮 최적화 모형 찾기 ")
use_quadratic = st.checkbox("비선형 같이 비교하기 (deg=2)", value=False)

results = []
for xcol, label in [("평균온도","온도"), ("습도","습도"), ("광량","광량")]:
    if xcol in vdf.columns:
        xd = vdf[[xcol, "잎길이"]].dropna()
        if len(xd) >= 2:
            # 선형
            b1, b0 = np.polyfit(xd[xcol], xd["잎길이"], 1)  # slope, intercept
            r = xd[xcol].corr(xd["잎길이"])  # 피어슨 상관
            row = {
+                 "변수": label,
+                 "기울기(k) (얼마나 빨리 늘어나는가)": round(float(b1), 4),
+                 "y절편(c) (시작 값)": round(float(b0), 4),
+                 "함께 변하는 정도 r(상관)": round(float(r), 4)
+             }
            if use_quadratic and len(xd) >= 3:
                a2, a1, a0 = np.polyfit(xd[xcol], xd["잎길이"], 2)
                row.update({
                    "이차항계수 a2": round(float(a2), 6),
                    "일차항계수 a1": round(float(a1), 5),
                    "상수항 a0": round(float(a0), 4)
                })
            results.append(row)

if results:
    st.dataframe(pd.DataFrame(results))
else:
    st.info("모형을 적합하기에 데이터가 부족합니다. 각 변수-잎길이 쌍에 최소 2개 이상의 데이터가 필요합니다.")

# ------------------------------
# Prediction sandbox
# ------------------------------
st.subheader("🔮 성장량 예측하기(선형모형, 직선식)")
colA, colB, colC, colD = st.columns(4)
with colA:
    T_in = st.number_input("예측용 온도(℃)", value=float(vdf["평균온도"].median()) if "평균온도" in vdf else 24.0)
with colB:
    H_in = st.number_input("예측용 습도(%)", value=float(vdf["습도"].median()) if "습도" in vdf else 60.0)
with colC:
    L_in = st.number_input("예측용 광량(lx)", value=float(vdf["광량"].median()) if "광량" in vdf else 20000.0)
with colD:
    base = float(vdf["잎길이"].iloc[0]) if "잎길이" in vdf and not vdf.empty else 4.0
    base_len = st.number_input("현재 잎길이(cm)", value=base)

# 선형계수 딕셔너리 구성
coef = {"평균온도": (0.0, 0.0), "습도": (0.0, 0.0), "광량": (0.0, 0.0)}
for res, xcol in zip(results, ["평균온도", "습도", "광량"]):
    if res["변수"] == "온도":
        coef["평균온도"] = (res["선형 기울기(k)"], res["절편(c)"])
    elif res["변수"] == "습도":
        coef["습도"] = (res["선형 기울기(k)"], res["절편(c)"])
    elif res["변수"] == "광량":
        coef["광량"] = (res["선형 기울기(k)"], res["절편(c)"])

# 단변수 선형모형을 가중 평균처럼 합성 (단순한 휴리스틱)
weights = {"평균온도": 1.0, "습도": 0.6, "광량": 0.8}
num = 0.0
w_sum = 0.0
for x, xval in zip(["평균온도","습도","광량"], [T_in, H_in, L_in]):
    k, c = coef.get(x, (0.0, 0.0))
    if k != 0.0 or c != 0.0:
        pred = k * xval + c
        num += weights[x] * pred
        w_sum += weights[x]

if w_sum > 0:
    blended = num / w_sum
    st.success(f"예상 잎길이: **{blended:.2f} cm** (by 예측 모델)")
else:
     st.info("아직 직선식이 만들어지지 않아 예측을 하지 않았어요. 위의 표를 먼저 만들어 주세요.")

# ------------------------------
# Notes & Tips
# ------------------------------
with st.expander("📌 팁: 데이터 수집/정제/분석 체크리스트"):
    st.markdown("""
    - 날짜는 YYYY-MM-DD 형식을 권장합니다.
    - 센서 단위: 온도(℃), 습도(%), 광량(lx)
    - 누락값(빈칸)은 제거되거나 0으로 처리되지 않도록 주의하세요.
    - 같은 날짜에 여러 식물 데이터를 기록해도 됩니다(식물 컬럼으로 구분).
    - 모델이 선형으로 안 맞으면, 이차모형 옵션을 켜고 비교해 보세요.
    """)
