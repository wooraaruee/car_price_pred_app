import streamlit as st
import pickle
import pandas as pd

# โหลดโมเดล
with open("C:/Users/HP/Desktop/web_app/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔮  Car Price Prediction")
st.write("กรอกข้อมูลรถของคุณเพื่อทำนายราคา (USD)")

# -------------------
# Input numerical features (float)
# -------------------
mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=35.0, value=15.0)
engine_cc = st.number_input("Engine (cc)", min_value=800.0, max_value=5000.0, value=1500.0)
owner_count = st.number_input("Owner Count", min_value=1.0, max_value=5.0, value=1.0)
accidents_reported = st.number_input("Accidents Reported", min_value=0.0, max_value=5.0, value=0.0)
make_year = st.number_input("Make year", min_value=1995.0, max_value=2023.0, value=2023.0)

# -------------------
# Input categorical features (string)
# -------------------
insurance_valid_input = st.selectbox("Insurance Valid", ['Yes', 'No'])
service_history = st.selectbox("Service History", ["None", "Full", "Partial"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
brand = st.selectbox("Brand", ['Chevrolet','Honda','BMW','Hyundai','Nissan','Tesla','Toyota','Kia','Volkswagen','Ford'])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
color = st.selectbox("Color", ['White','Black','Blue','Red','Gray','Silver'])

# แปลง categorical ให้เป็น string
insurance_valid = insurance_valid_input
car_age = 2025.0 - make_year  # numeric feature

# -------------------
# Predict
# -------------------
if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        mileage_kmpl, engine_cc, owner_count, insurance_valid,
        accidents_reported, service_history, fuel_type,
        brand, transmission, color, car_age
    ]], columns=[
        'mileage_kmpl','engine_cc','owner_count','insurance_valid',
        'accidents_reported','service_history','fuel_type',
        'brand','transmission','color','Car_Age'
    ])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 ราคาประมาณ: {prediction:,.2f} USD")