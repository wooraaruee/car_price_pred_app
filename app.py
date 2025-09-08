import streamlit as st

# ตั้งค่า page
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction App")
st.write("ยินดีต้อนรับสู่แอปทำนายราคารถยนต์ 🚘")
st.write("แอปนี้เป็นตัวอย่างสำหรับการศึกษาเท่านั้น **ไม่สามารถใช้ทำนายราคาจริงได้**")

st.markdown("""
### วิธีใช้งาน
- คลิก **Go to Predict** เพื่อกรอกข้อมูลรถและทำนายราคา
- คลิก **About** เพื่อดูรายละเอียดของแอปและข้อมูลเพิ่มเติม
""")
