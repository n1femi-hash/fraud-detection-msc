import streamlit as st
from datetime import datetime

st.title("💳 Fraud Detection Input Form")

# === Transaction Details ===
with st.expander("📅 Transaction Details", expanded=True):
    trans_date = st.date_input("Transaction Date", value=datetime(2024, 5, 15))
    trans_time = st.time_input("Transaction Time", value=datetime(2024, 5, 15, 14, 30).time())
    trans_datetime = datetime.combine(trans_date, trans_time)

    cc_num = st.text_input("Credit Card Number", value="1234567890123456")
    merchant = st.text_input("Merchant Name/ID", value="ABC Store")
    category = st.text_input("Transaction Category", value="shopping_pos")
    amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01, value=75.50)

# === Cardholder Details ===
with st.expander("🧑 Cardholder Details", expanded=True):
    first = st.text_input("First Name", value="John")
    last = st.text_input("Last Name", value="Doe")
    gender = st.selectbox("Gender", options=["M", "F", "Other"], index=0)
    street = st.text_input("Street Address", value="123 Main St")
    city = st.text_input("City", value="New York")
    state = st.text_input("State", value="NY")
    zip_code = st.text_input("ZIP Code", value="10001")

    lat = st.number_input("Latitude", format="%.6f", value=40.712776)
    long = st.number_input("Longitude", format="%.6f", value=-74.005974)
    city_pop = st.number_input("City Population", min_value=0, value=8419600)

    job = st.text_input("Job Title", value="Engineer")
    dob = st.date_input("Date of Birth", value=datetime(1990, 1, 1))

# === Merchant Details ===
with st.expander("🏪 Merchant Details", expanded=True):
    trans_num = st.text_input("Transaction Number", value="T123456789")
    unix_time = int(trans_datetime.timestamp())
    merch_lat = st.number_input("Merchant Latitude", format="%.6f", value=40.730610)
    merch_long = st.number_input("Merchant Longitude", format="%.6f", value=-73.935242)

# === Submit ===
if st.button("✅ Submit"):
    input_record = {
        "trans_date_trans_time": trans_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cc_num,
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "first": first,
        "last": last,
        "gender": gender,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "dob": dob.strftime("%Y-%m-%d"),
        "trans_num": trans_num,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
    }
    st.success("✅ Form Submitted!")
    st.json(input_record)
