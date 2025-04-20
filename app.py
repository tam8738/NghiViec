import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load mô hình và scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("🔍 Dự đoán nhân viên có nghỉ việc không")

# Giao diện nhập liệu
st.header("Nhập thông tin nhân viên:")

mucdohailong = st.slider("Mức độ hài lòng", 0.0, 1.0, 0.5)
danhgia_gannhat = st.slider("Đánh giá gần nhất", 0.0, 1.0, 0.5)
soluong_duan = st.number_input("Số lượng dự án tham gia", 1, 10, 3)
sogiolam = st.number_input("Số giờ làm trung bình hàng tháng", 90, 350, 160)
sonam = st.number_input("Số năm làm việc", 1, 15, 3)
thangchuc = st.selectbox("Được thăng chức trong 5 năm?", [0, 1])
mucluong = st.selectbox("Mức lương", ['low', 'medium', 'high'])
phongban = st.selectbox("Phòng ban", [
    'accounting', 'hr', 'IT', 'management', 'marketing', 
    'product_mng', 'RandD', 'sales', 'support', 'technical'
])

# Mã hóa mucluong và phongban
mucluong_map = {'low': 0, 'medium': 1, 'high': 2}
mucluong_encoded = mucluong_map[mucluong]

# One-hot encoding cho phongban
phongban_cols = [name for name in feature_names if name.startswith('phongban_')]
phongban_dict = dict.fromkeys(phongban_cols, 0)

col_name = f'phongban_{phongban}'
if col_name in phongban_cols:
    phongban_dict[col_name] = 1

# Tạo dataframe đầu vào
input_data = {
    'mucdohailong': mucdohailong,
    'danhgia_gannhat': danhgia_gannhat,
    'soluong_duanthamgia': soluong_duan,
    'sogiolamtrungbinh_hangthang': sogiolam,
    'sonamlamviec': sonam,
    'duocthangchuc_trong5nam': thangchuc,
    'mucluong': mucluong_encoded
}
input_data.update(phongban_dict)

X_input = pd.DataFrame([input_data])
X_input_scaled = scaler.transform(X_input)

# Dự đoán
if st.button("Dự đoán"):
    prediction = model.predict(X_input_scaled)[0]
    probabilities = model.predict_proba(X_input_scaled)[0]
    pct_no = probabilities[0] * 100
    pct_yes = probabilities[1] * 100

    st.subheader("🔎 Kết quả:")
    if prediction == 1:
        st.error(f"Nhân viên CÓ thể nghỉ việc ({pct_yes:.2f}%)")
    else:
        st.success(f"Nhân viên KHÔNG nghỉ việc ({pct_no:.2f}%)")

    st.write(f"Xác suất KHÔNG nghỉ việc: **{pct_no:.2f}%**")
    st.write(f"Xác suất CÓ thể nghỉ việc: **{pct_yes:.2f}%**")
