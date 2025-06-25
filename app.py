import streamlit as st
import numpy as np
import joblib

# Load model đã train sẵn
model = joblib.load("diabetes_model.pkl")  # hoặc model bạn đã train bằng sklearn

# Tiêu đề app
st.title("Dự đoán bệnh tiểu đường 🚑")

# Tạo các ô nhập liệu
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=33)

# Nút dự đoán
if st.button("Dự đoán"):
    # Tạo mảng input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Thực hiện dự đoán
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Bạn có nguy cơ bị tiểu đường.")
    else:
        st.success("✅ Bạn KHÔNG có nguy cơ bị tiểu đường.")

