from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="การจำแนกโรคหัวใจด้วย KNN",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Minimalist Dark-Navy Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sarabun', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base Background */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
}

/* Hide Default Streamlit Chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stDecoration"] {display: none;}

/* Header Card */
.header-box {
    background: linear-gradient(180deg, #101726 0%, #0d1322 100%);
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.header-title {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 600;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

/* Section Banner */
.section-banner {
    background-color: #111a2e;
    border-left: 3px solid #2563eb;
    border-top: 1px solid #1a2744;
    border-right: 1px solid #1a2744;
    border-bottom: 1px solid #1a2744;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 24px 0 16px 0;
}
.section-banner-title {
    color: #e2e8f0;
    font-size: 1.02rem;
    font-weight: 500;
    margin: 0;
}

/* Preview Image Cards */
.image-card-title {
    color: #cbd5e1;
    font-size: 0.95rem;
    font-weight: 500;
    padding: 10px 14px;
    background-color: #101726;
    border: 1px solid #1e2d4a;
    border-bottom: 1px solid #162238;
    border-radius: 8px 8px 0 0;
    text-align: center;
}

/* Uniform Image Display for perfectly matched sizes */
[data-testid="stImage"] {
    width: 100% !important;
}

[data-testid="stImage"] > img {
    width: 100% !important;
    height: 300px !important;
    object-fit: cover !important;
    border-radius: 0 0 8px 8px !important;
    border: 1px solid #1e2d4a !important;
    border-top: none !important;
    display: block !important;
}

/* Metrics Styling */
[data-testid="stMetric"] {
    background-color: #101726;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #0d1322 !important;
    color: #f1f5f9 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 6px !important;
}

/* Result Cards */
.result-card-positive {
    background-color: #141b2b;
    border: 1px solid #991b1b;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 18px 22px;
    margin-top: 16px;
    margin-bottom: 16px;
}
.result-card-negative {
    background-color: #141b2b;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 18px 22px;
    margin-top: 16px;
    margin-bottom: 16px;
}
.result-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.result-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    line-height: 1.5;
    margin: 0;
}

/* Input Fields & Widgets */
.stNumberInput > div > div > input {
    background-color: #0d1322 !important;
    color: #f1f5f9 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 6px !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}

/* Buttons */
div.stButton > button {
    background-color: #1d4ed8;
    color: #ffffff;
    border: 1px solid #2563eb;
    border-radius: 6px;
    padding: 10px 24px;
    font-weight: 500;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2d4a;
    border-radius: 6px;
    background-color: #101726;
}

/* Minimal Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0b0f19;
}
::-webkit-scrollbar-thumb {
    background: #1e2d4a;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #2563eb;
}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-box">
    <div class="header-title">ระบบจำแนกโรคหัวใจด้วยอัลกอริทึม K-Nearest Neighbors</div>
    <div class="header-subtitle">การวิเคราะห์และประเมินความเสี่ยงโรคหัวใจจากชุดข้อมูลสุขภาพทางคลินิก</div>
</div>
""", unsafe_allow_html=True)

# Overview Cards
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="image-card-title">กลุ่มตัวอย่าง: พบภาวะโรคหัวใจ</div>', unsafe_allow_html=True)
    st.image("./img/Disease.jpg")

with col2:
    st.markdown('<div class="image-card-title">กลุ่มตัวอย่าง: ไม่พบภาวะโรคหัวใจ</div>', unsafe_allow_html=True)
    st.image("./img/UnDisease.jpg")

# Dataset Section
st.markdown("""
<div class="section-banner">
    <div class="section-banner-title">ชุดข้อมูลตัวอย่าง (Heart Disease Dataset)</div>
</div>
""", unsafe_allow_html=True)

dt = pd.read_csv("./data/Heart3.csv")
st.dataframe(dt.head(10))

# Data Visualization Section
show_viz = st.checkbox("แสดงแผนภูมิวิเคราะห์ข้อมูลทางสถิติ", value=False)

if show_viz:
    # 3 Summary Metrics in Minimalist Dark-Navy Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    total_records = len(dt)
    normal_count = int((dt['HeartDisease'] == 0).sum())
    disease_count = int((dt['HeartDisease'] == 1).sum())
    normal_pct = (normal_count / total_records) * 100
    disease_pct = (disease_count / total_records) * 100

    with m_col1:
        st.metric("จำนวนกลุ่มตัวอย่างทั้งหมด", f"{total_records:,} ราย")
    with m_col2:
        st.metric("สุขภาพปกติ (ไม่พบโรค)", f"{normal_count:,} ราย", f"{normal_pct:.1f}%")
    with m_col3:
        st.metric("พบความเสี่ยงโรคหัวใจ", f"{disease_count:,} ราย", f"{disease_pct:.1f}%")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    viz_choice = st.selectbox(
        "เลือกมุมมองแผนภูมิข้อมูลที่ต้องการวิเคราะห์:",
        [
            "เปรียบเทียบค่าเฉลี่ยตัวแปรสุขภาพสำคัญ (ปกติ vs พบโรคหัวใจ)",
            "สัดส่วนปัจจัยเสี่ยงทางคลินิก (%)",
            "สัดส่วนจำนวนผู้ป่วยตามผลการวินิจฉัย (Distribution)"
        ]
    )

    if viz_choice == "เปรียบเทียบค่าเฉลี่ยตัวแปรสุขภาพสำคัญ (ปกติ vs พบโรคหัวใจ)":
        metrics_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
        mean_df = dt.groupby('HeartDisease')[metrics_cols].mean().round(1).T
        mean_df.columns = ["ไม่พบโรคหัวใจ", "พบความเสี่ยงโรคหัวใจ"]
        mean_df.index = ["อายุเฉลี่ย (ปี)", "ความดันโลหิตเฉลี่ย (mmHg)", "คอเลสเตอรอลเฉลี่ย (mg/dl)", "อัตราเต้นหัวใจสูงสุด (bpm)"]
        st.bar_chart(mean_df)
        st.caption("เปรียบเทียบค่าเฉลี่ยตัวแปรสรีรวิทยาสำคัญ สเกลข้อมูลมีความสอดคล้องและสัมพันธ์กัน")

    elif viz_choice == "สัดส่วนปัจจัยเสี่ยงทางคลินิก (%)":
        risk_male = (dt.groupby('HeartDisease')['Sex'].mean() * 100).round(1)
        risk_fbs = (dt.groupby('HeartDisease')['FastingBS'].mean() * 100).round(1)
        risk_angina = (dt.groupby('HeartDisease')['ExerciseAngina'].mean() * 100).round(1)

        risk_df = pd.DataFrame({
            "ไม่พบโรคหัวใจ": [risk_male[0], risk_fbs[0], risk_angina[0]],
            "พบความเสี่ยงโรคหัวใจ": [risk_male[1], risk_fbs[1], risk_angina[1]]
        }, index=["เพศชาย (%)", "น้ำตาลในเลือดสูง >120 mg/dl (%)", "แน่นหน้าอกจากการออกกำลังกาย (%)"])
        st.bar_chart(risk_df)
        st.caption("สัดส่วนร้อยละของปัจจัยเสี่ยงทางคลินิกในแต่ละกลุ่ม (สเกลมาตรฐาน 0 - 100%)")

    else:
        target_counts = dt['HeartDisease'].value_counts().rename(index={0: "ไม่พบโรคหัวใจ", 1: "พบความเสี่ยงโรคหัวใจ"})
        target_df = pd.DataFrame({"จำนวนกลุ่มตัวอย่าง (ราย)": target_counts})
        st.bar_chart(target_df)
        st.caption("สัดส่วนจำนวนกลุ่มตัวอย่างทั้งหมด 918 ราย จำแนกตามผลการวินิจฉัย")

# Prediction Form Section
st.markdown("""
<div class="section-banner">
    <div class="section-banner-title">การระบุข้อมูลเพื่อประเมินความเสี่ยง</div>
</div>
""", unsafe_allow_html=True)

pt_age = st.slider("อายุ (Age)", min_value=0, max_value=100, value=40)

input_col1, input_col2 = st.columns(2)

with input_col1: 
    pt_sex = st.number_input("เพศ (Sex: 1 = ชาย, 0 = หญิง)", min_value=0, max_value=1, value=1)
    sp_ChestPainType = st.number_input("ประเภทอาการเจ็บหน้าอก (ChestPainType: 0 - 3)", min_value=0, value=1)
    sp_RestingBP = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=0, value=120)
    sp_Cholesterol = st.number_input("ระดับคอเลสเตอรอล (Cholesterol)", min_value=0, value=200)
    sp_FastingBS = st.number_input("ระดับน้ำตาลในเลือดหลังอดอาหาร (FastingBS: 0 หรือ 1)", min_value=0, max_value=1, value=0)

with input_col2:
    sp_RestingECG = st.number_input("คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)", min_value=0, value=0)
    sp_MaxHR = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=0, value=150)
    sp_ExerciseAngina = st.number_input("อาการเจ็บหน้าอกจากออกกำลังกาย (ExerciseAngina: 0 หรือ 1)", min_value=0, max_value=1, value=0)
    sp_Oldpeak = st.number_input("ระดับ ST Depression (Oldpeak)", min_value=0.0, step=0.1, value=0.0)
    sp_ST_Slope = st.number_input("ความชัน ST Segment (ST_Slope)", min_value=0, value=1)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

predict_clicked = st.button("ประเมินผลการทำนาย")

if predict_clicked:
    X = dt.drop('HeartDisease', axis=1)
    y = dt.HeartDisease   
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X, y)  

    x_input = np.array([[pt_age, pt_sex, sp_ChestPainType, sp_RestingBP, sp_Cholesterol, 
                         sp_FastingBS, sp_RestingECG, sp_MaxHR, sp_ExerciseAngina, 
                         sp_Oldpeak, sp_ST_Slope]])
    
    prediction = knn_model.predict(x_input)
    result = int(prediction[0])

    res_col1, res_col2 = st.columns([3, 2])
    
    with res_col1:
        if result == 1:
            st.markdown("""
            <div class="result-card-positive">
                <div class="result-title" style="color: #f87171;">ผลการประเมิน: มีความเสี่ยงเป็นโรคหัวใจ (Positive)</div>
                <p class="result-subtitle">แบบจำลอง KNN ตรวจพบความเสี่ยงจากข้อมูลตัวแปรสุขภาพที่ป้อนเข้าระบบ ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อรับการวินิจฉัยอย่างละเอียด</p>
            </div>
            """, unsafe_allow_html=True)
            st.image("./img/Disease.jpg", caption="ภาพประกอบ: พบภาวะโรคหัวใจ")
        else:
            st.markdown("""
            <div class="result-card-negative">
                <div class="result-title" style="color: #60a5fa;">ผลการประเมิน: ปกติ ไม่พบความเสี่ยงโรคหัวใจ (Negative)</div>
                <p class="result-subtitle">แบบจำลอง KNN ไม่พบความผิดปกติที่สอดคล้องกับภาวะโรคหัวใจจากข้อมูลตัวแปรสุขภาพที่ระบุ</p>
            </div>
            """, unsafe_allow_html=True)
            st.image("./img/UnDisease.jpg", caption="ภาพประกอบ: ไม่พบภาวะโรคหัวใจ")
