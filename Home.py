from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import base64
import os

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

st.set_page_config(
    page_title="การจำแนกโรคหัวใจด้วย KNN",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to load base64 image
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

ohm_img_b64 = get_base64_image("./img/Ohm.jpg")
if not ohm_img_b64:
    ohm_img_b64 = get_base64_image("./img/Ohm.webp")

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

/* Header Card with Top-Right Profile */
.header-box {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(180deg, #101726 0%, #0d1322 100%);
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 24px;
    gap: 20px;
    flex-wrap: wrap;
}
.header-left {
    flex: 1;
    min-width: 320px;
}
.header-title {
    color: #f8fafc;
    font-size: 1.6rem;
    font-weight: 600;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

/* Perfectly Fitted Student Profile Card (Top-Right) */
.student-profile-card {
    display: flex;
    align-items: center;
    gap: 16px;
    background-color: #0b0f19;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 18px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
}
.student-info {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: right;
    gap: 4px;
}
.student-name {
    color: #ffffff;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    line-height: 1.3;
}
.student-id {
    color: #60a5fa;
    font-size: 0.96rem;
    font-weight: 500;
    line-height: 1.3;
}
.student-dept {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.3;
}
.student-img-wrapper {
    width: 96px;
    height: 96px;
    min-width: 96px;
    min-height: 96px;
    max-width: 96px;
    max-height: 96px;
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid #2563eb;
    background-color: #0b0f19;
    flex-shrink: 0;
    box-shadow: 0 0 12px rgba(37, 99, 235, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    margin: 0;
}
.student-img-wrapper img.student-full-img {
    width: 100% !important;
    height: 100% !important;
    min-width: 100% !important;
    min-height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: cover !important;
    object-position: center center !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 6px !important;
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

/* Chart Container Header */
.chart-card-header {
    color: #cbd5e1;
    font-size: 0.96rem;
    font-weight: 500;
    margin-bottom: 8px;
    text-align: center;
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

# Header Section with Perfectly Fitted Profile Card at Top-Right
header_html = f"""
<div class="header-box">
    <div class="header-left">
        <div class="header-title">ระบบจำแนกโรคหัวใจด้วยอัลกอริทึม K-Nearest Neighbors</div>
        <div class="header-subtitle">การวิเคราะห์และประเมินความเสี่ยงโรคหัวใจจากชุดข้อมูลสุขภาพทางคลินิก</div>
    </div>
    <div class="student-profile-card">
        <div class="student-info">
            <div class="student-name">นาย ปรเมศ ไม้ประเสริฐ</div>
            <div class="student-id">รหัสนักศึกษา: 674230020</div>
            <div class="student-dept">สาขาเทคโนโลยีสารสนเทศ</div>
        </div>
        <div class="student-img-wrapper">
            <img src="data:image/jpeg;base64,{ohm_img_b64}" class="student-full-img" alt="นาย ปรเมศ ไม้ประเสริฐ">
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

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
show_viz = st.checkbox("แสดงแผนภูมิวิเคราะห์ข้อมูลทางสถิติ", value=True)

if show_viz:
    total_records = len(dt)
    normal_count = int((dt['HeartDisease'] == 0).sum())
    disease_count = int((dt['HeartDisease'] == 1).sum())
    normal_pct = (normal_count / total_records) * 100
    disease_pct = (disease_count / total_records) * 100

    # 3 Summary Metrics in Minimalist Dark-Navy Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("จำนวนกลุ่มตัวอย่างทั้งหมด", f"{total_records:,} ราย")
    with m_col2:
        st.metric("สุขภาพปกติ (ไม่พบโรค)", f"{normal_count:,} ราย", f"{normal_pct:.1f}%")
    with m_col3:
        st.metric("พบความเสี่ยงโรคหัวใจ", f"{disease_count:,} ราย", f"{disease_pct:.1f}%")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 2 Charts Side-by-Side: Donut Chart & Grouped Bar Chart
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="chart-card-header">แผนภาพวงกลม: สัดส่วนผลการวินิจฉัยโรคหัวใจ</div>', unsafe_allow_html=True)
        pie_labels = ["สุขภาพปกติ", "พบความเสี่ยงโรคหัวใจ"]
        pie_values = [normal_count, disease_count]
        pie_colors = ['#38bdf8', '#1d4ed8']

        if HAS_PLOTLY:
            fig_pie = go.Figure(data=[go.Pie(
                labels=pie_labels,
                values=pie_values,
                hole=0.58,
                marker=dict(colors=pie_colors, line=dict(color='#0b0f19', width=2)),
                textinfo='percent+label',
                textposition='inside',
                textfont=dict(size=12, color='#ffffff'),
                hoverinfo='label+value+percent'
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#94a3b8', size=11)
                ),
                margin=dict(t=15, b=35, l=15, r=15),
                height=310
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        else:
            fig, ax = plt.subplots(figsize=(5, 3.8), facecolor='#101726')
            ax.set_facecolor('#101726')
            wedges, texts, autotexts = ax.pie(
                pie_values,
                labels=pie_labels,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.45, edgecolor='#0b0f19', linewidth=2),
                textprops=dict(color='#f1f5f9', size=10)
            )
            for at in autotexts:
                at.set_color('#ffffff')
            st.pyplot(fig)

    with chart_col2:
        st.markdown('<div class="chart-card-header">แผนภูมิแท่ง: เปรียบเทียบค่าเฉลี่ยตัวแปรสำคัญ</div>', unsafe_allow_html=True)
        bar_metrics = ["อายุ (ปี)", "ความดัน (mmHg)", "คอเลสเตอรอล", "อัตราเต้นหัวใจ"]
        
        # Calculate precise means
        sub_norm = dt[dt['HeartDisease'] == 0]
        sub_dise = dt[dt['HeartDisease'] == 1]
        
        val_norm = [
            round(sub_norm['Age'].mean(), 1),
            round(sub_norm['RestingBP'].mean(), 1),
            round(sub_norm['Cholesterol'].mean(), 1),
            round(sub_norm['MaxHR'].mean(), 1)
        ]
        val_dise = [
            round(sub_dise['Age'].mean(), 1),
            round(sub_dise['RestingBP'].mean(), 1),
            round(sub_dise['Cholesterol'].mean(), 1),
            round(sub_dise['MaxHR'].mean(), 1)
        ]

        if HAS_PLOTLY:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bar_metrics,
                y=val_norm,
                name="สุขภาพปกติ",
                marker_color='#38bdf8',
                text=val_norm,
                textposition='outside',
                textfont=dict(color='#cbd5e1', size=11)
            ))
            fig_bar.add_trace(go.Bar(
                x=bar_metrics,
                y=val_dise,
                name="พบความเสี่ยง",
                marker_color='#1d4ed8',
                text=val_dise,
                textposition='outside',
                textfont=dict(color='#cbd5e1', size=11)
            ))
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(color='#e2e8f0', size=11),
                    linecolor='#1e2d4a'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#162238',
                    tickfont=dict(color='#94a3b8', size=10),
                    linecolor='#1e2d4a'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.28,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#94a3b8', size=11)
                ),
                margin=dict(t=15, b=45, l=15, r=15),
                height=310
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        else:
            fig, ax = plt.subplots(figsize=(5, 3.8), facecolor='#101726')
            ax.set_facecolor('#101726')
            x = np.arange(len(bar_metrics))
            width = 0.35
            rects1 = ax.bar(x - width/2, val_norm, width, label='สุขภาพปกติ', color='#38bdf8')
            rects2 = ax.bar(x + width/2, val_dise, width, label='พบความเสี่ยง', color='#1d4ed8')
            ax.set_xticks(x)
            ax.set_xticklabels(bar_metrics, color='#e2e8f0', size=9)
            ax.tick_params(colors='#94a3b8')
            ax.legend(facecolor='#101726', edgecolor='#1e2d4a', labelcolor='#e2e8f0')
            ax.bar_label(rects1, padding=2, color='#cbd5e1', size=8)
            ax.bar_label(rects2, padding=2, color='#cbd5e1', size=8)
            st.pyplot(fig)

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
