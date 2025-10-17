import streamlit as st
import pandas as pd
from src.data_processor import clean_and_engineer_features
from src.model_basic import train_model

@st.cache_data(show_spinner=False)
def load_model():
    model = train_model("data/Bengaluru_House_Data.csv")
    return model

def format_price(price_lakhs):
    if price_lakhs >= 100:
        crore = price_lakhs / 100
        return f"{crore:.2f} Crore"
    return f"{price_lakhs:.2f} Lakhs"

def predict_price(model, area_type, location, total_sqft, bath, bhk):
    input_df = pd.DataFrame({
        "area_type": [area_type],
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "bhk": [bhk],
        "size": [f"{bhk} BHK"],
        "price": [0]
    })

    input_df_clean = clean_and_engineer_features(input_df)
    if input_df_clean.empty:
        return "Sorry, this combination of inputs is not supported."
    X = input_df_clean.drop(columns=["price"])
    predicted_price = model.predict(X)[0]
    return format_price(predicted_price)

def main():
    st.title("Bengaluru House Price Prediction")
    st.write("Enter the house details and get the estimated price!")

    model = load_model()

    area_type_options = [
        "Super built-up  Area", 
        "Built-up  Area", 
        "Plot  Area", 
        "Carpet  Area"
    ]
    area_type = st.selectbox("Area Type", area_type_options)

    location_options = [
        "Whitefield", "Sarjapur Road", "Electronic City", "Marathahalli", "Koramangala"
    ]
    location = st.selectbox("Location", location_options)

    total_sqft = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, value=1000.0)
    bath = st.slider("Number of Bathrooms", 1, 5, 2)
    bhk = st.slider("Number of BHK (Bedrooms)", 1, 5, 3)

    if st.button("Predict Price"):
        result = predict_price(model, area_type, location, total_sqft, bath, bhk)
        st.success(f"Estimated House Price: {result}")

if __name__ == "__main__":
    main()
