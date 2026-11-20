from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกโรคหัวใจด้วย KNN")
#st.image("./img/kairung022.jpg")
col1, col2 = st.columns(2)

with col1:
   st.header("เป็นโรคหัวใจ")
   st.image("./img/Disease.jpg")

with col2:
   st.header("ไม่เป็นโรคหัวใจ")
   st.image("./img/UnDisease.jpg")


html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Age'].sum()


dt2 = dt['Sex'].sum()
dt3 = dt['ChestPainType'].sum()
dt4 = dt['RestingBP'].sum()
dt5 = dt['Cholesterol'].sum()
dt6 = dt['FastingBS'].sum()
dt7 = dt['RestingECG'].sum()
dt8 = dt['MaxHR'].sum()
dt9 = dt['ExerciseAngina'].sum()
dt10 = dt['Oldpeak'].sum()
dt11 = dt['ST_Slope'].sum()

dx = [dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10, dt11]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11" ])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

input_col1, input_col2 = st.columns(2)

with input_col1:
   pt_age = st.slider("กรุณาเลือกข้อมูล Age", min_value=0, max_value=100)
   pt_sex = st.number_input("กรุณาเลือกข้อมูล sex", min_value=0, max_value=1)
   sp_ChestPainType = st.number_input("กรุณาเลือกข้อมูล ChestPainType", min_value=0)
   sp_RestingBP = st.number_input("กรุณาเลือกข้อมูล RestingBP", min_value=0)
   sp_Cholesterol = st.number_input("กรุณาเลือกข้อมูล Cholesterol", min_value=0)
   sp_FastingBS = st.number_input("กรุณาเลือกข้อมูล FastingBS", min_value=0, max_value=1)

with input_col2:
   sp_RestingECG = st.number_input("กรุณาเลือกข้อมูล RestingECG", min_value=0)
   sp_MaxHR = st.number_input("กรุณาเลือกข้อมูล MaxHR", min_value=0)
   sp_ExerciseAngina = st.number_input("กรุณาเลือกข้อมูล ExerciseAngina", min_value=0, max_value=1)
   sp_Oldpeak = st.number_input("กรุณาเลือกข้อมูล Oldpeak", min_value=0)
   sp_ST_Slope = st.number_input("กรุณาเลือกข้อมูล ST_Slope", min_value=0)

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris.csv")   

   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease   
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[pt_age, pt_sex, sp_ChestPainType, sp_RestingBP, sp_Cholesterol, sp_FastingBS, sp_RestingECG, sp_MaxHR, sp_ExerciseAngina, sp_Oldpeak, sp_ST_Slope]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

   if out == 1:
    st.image("./img/Disease.jpg", caption="ผลทำนาย: พบโรคหัวใจ")
   else:
    st.image("./img/UnDisease.jpg", caption="ผลทำนาย: ไม่พบโรคหัวใจ")
else:
    st.write("ไม่ทำนาย")
