import pickle
import pandas as pd
import streamlit as st

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
@st.cache_resource
def load_model():
    with open("predicted_sales_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()

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

brand_name = st.selectbox("Brand Name", brand_options)
material_family = st.selectbox("Material Family", material_options)

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