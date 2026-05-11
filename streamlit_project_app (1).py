import pickle
import pandas as pd
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Predicted Sales Quantity",
    layout="wide"
)

st.title("📊 Predicted Sales Quantity")
st.write(
    "This app predicts the quantity of products needed for a specific day, month, and year."
)

# -----------------------------
# Load model
# -----------------------------
model_data = pickle.load(open("predicted_sales_model.pkl", "rb"))

model = model_data["model"]
features = model_data["features"]
brand_options = model_data["brand_options"]
material_options = model_data["material_options"]

# -----------------------------
# User inputs
# -----------------------------
st.subheader("Enter prediction details:")

selected_date = st.date_input("Select Date")

day = selected_date.day
month = selected_date.month
year = selected_date.year
day_of_week = selected_date.weekday()
quarter = ((month - 1) // 3) + 1

brand_name = st.selectbox("Brand Name", brand_options)
material_family = st.selectbox("Material Family", material_options)

st.subheader("Historical Sales Inputs")

prev_qty = st.number_input("Previous Quantity", value=0.0)
prev_qty_7 = st.number_input("Quantity 7 Days Ago", value=0.0)
prev_qty_14 = st.number_input("Quantity 14 Days Ago", value=0.0)
prev_qty_30 = st.number_input("Quantity 30 Days Ago", value=0.0)

rolling_avg_7 = st.number_input("7-Day Rolling Average", value=0.0)
rolling_avg_14 = st.number_input("14-Day Rolling Average", value=0.0)
rolling_avg_30 = st.number_input("30-Day Rolling Average", value=0.0)

expanding_mean = st.number_input("Historical Average Quantity", value=0.0)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Quantity"):

    input_data = {
        "day": day,
        "month": month,
        "year": year,
        "Brand Name": brand_name,
        "Material Family": material_family
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted product quantity: {round(prediction, 0)} units")

    st.write("Input used for prediction:")
    st.dataframe(input_df)