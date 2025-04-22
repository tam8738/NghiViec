import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load mô hình và scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Dự đoán nghỉ việc", layout="centered")

st.markdown("## Dự đoán khả năng nghỉ việc của nhân viên")
st.markdown("Nhập thông tin bên dưới để hệ thống dự đoán khả năng nghỉ việc.")

# ======= Nhập thông tin =======
with st.form("form_du_doan"):
    col1, col2 = st.columns(2)

    with col1:
        mucdohailong = st.slider("Mức độ hài lòng (0 ➝ 1)", 0.0, 1.0, 0.62, step=0.01)
        danhgia_gannhat = st.slider("Đánh giá gần nhất (0 ➝ 1)", 0.0, 1.0, 0.5, step=0.01)
        soluong_duan = st.number_input("Số dự án đã tham gia", 1, 10, 3)
        sogiolam = st.number_input("Số giờ làm trung bình/tháng", 90, 350, 160)

    with col2:
        sonam = st.number_input("Số năm làm việc", 1, 15, 3)
        thangchuc = st.selectbox("Được thăng chức trong 5 năm?", ["Không", "Có"])
        mucluong = st.selectbox("Mức lương", ['low', 'medium', 'high'])
        phongban = st.selectbox("Phòng ban", [
            'accounting', 'hr', 'IT', 'management', 'marketing',
            'product_mng', 'RandD', 'sales', 'support', 'technical'
        ])

    submitted = st.form_submit_button(" Dự đoán")

# ======= Xử lý dữ liệu =======
if submitted:
    thangchuc_bin = 1 if thangchuc == "Có" else 0
    mucluong_map = {'low': 0, 'medium': 1, 'high': 2}
    mucluong_encoded = mucluong_map[mucluong]

    phongban_cols = [name for name in feature_names if name.startswith('phongban_')]
    phongban_dict = dict.fromkeys(phongban_cols, 0)
    col_name = f'phongban_{phongban}'
    if col_name in phongban_dict:
        phongban_dict[col_name] = 1

    input_data = {
        'mucdohailong': mucdohailong,
        'danhgia_gannhat': danhgia_gannhat,
        'soluong_duanthamgia': soluong_duan,
        'sogiolamtrungbinh_hangthang': sogiolam,
        'sonamlamviec': sonam,
        'duocthangchuc_trong5nam': thangchuc_bin,
        'mucluong': mucluong_encoded
    }
    input_data.update(phongban_dict)

    X_input = pd.DataFrame([input_data])
    X_input_scaled = scaler.transform(X_input)

    prediction = model.predict(X_input_scaled)[0]
    probabilities = model.predict_proba(X_input_scaled)[0]
    pct_no = probabilities[0] * 100
    pct_yes = probabilities[1] * 100

    # ======= Hiển thị kết quả =======
    st.markdown("Kết quả dự đoán")
    if prediction == 1:
        st.error(f"⚠️ Nhân viên **CÓ thể nghỉ việc** ({pct_yes:.2f}%)")
    else:
        st.success(f" Nhân viên **KHÔNG nghỉ việc** ({pct_no:.2f}%)")

    st.markdown(f"- Xác suất **KHÔNG nghỉ việc**: `{pct_no:.2f}%`")
    st.markdown(f"- Xác suất **CÓ thể nghỉ việc**: `{pct_yes:.2f}%`")
