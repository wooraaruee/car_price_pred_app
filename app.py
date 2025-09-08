import streamlit as st

# ตั้งค่า page
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Welcome to Car Price Prediction App")
st.write("แอปนี้ช่วยคุณทำนายราคาของรถยนต์โดยใช้ Machine Learning")

st.markdown("""
- กดปุ่ม **Go to Predict** เพื่อกรอกข้อมูลรถและทำนายราคา  
- กดปุ่ม **About** เพื่อดูรายละเอียดแอป
""")
