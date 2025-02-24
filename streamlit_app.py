import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models
model = joblib.load("car_price_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the menu to navigate the app.")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Prediction", "About"])

if page == "Prediction":
    st.title("🚗 Car Price Prediction App")
    st.markdown("### Enter the details below to predict the present price of a car.")

    # User input with validation messages
    selling_price = st.text_input("Selling Price (in lakhs)", placeholder="e.g., 3.5 for 3.5 lakhs")
    driven_kms = st.text_input("Kilometers Driven", placeholder="e.g., 45000 for 45000 km")
    car_name = st.text_input("Car Name", placeholder="e.g., Maruti Swift, Hyundai i20")

    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"], index=None, placeholder="Select Fuel Type")
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"], index=None, placeholder="Select Selling Type")
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"], index=None, placeholder="Select Transmission")
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3], index=None, placeholder="Select Owner Count")
    car_age = st.text_input("Car Age (in years)", placeholder="e.g., 5 for 5 years")

    # Ensure all fields are filled before displaying the Predict button
    if not (selling_price and driven_kms and car_name and fuel_type and selling_type and transmission and owner is not None and car_age):
        st.warning("Please fill in all fields to enable the prediction button.")
    else:
        if st.button("Predict Present Price", key="predict_button"):
            try:
                selling_price = float(selling_price)
                driven_kms = int(driven_kms)
                owner = int(owner)
                car_age = int(car_age)
                
                data = pd.DataFrame([[selling_price, driven_kms, fuel_type, selling_type, transmission, owner, car_age, car_name]],
                                     columns=["Selling_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner", "Car_Age", "Car_Name"])
                
                # Rename Selling_Price to Present_Price to match model training
                data.rename(columns={"Selling_Price": "Present_Price"}, inplace=True)
                
                # Encode categorical variables
                data_encoded = encoder.transform(data[["Fuel_Type", "Selling_type", "Transmission"]])
                
                # Scale numerical variables
                data_scaled = scaler.transform(data.drop(columns=["Fuel_Type", "Selling_type", "Transmission", "Car_Name"]))
                
                # Combine encoded and scaled data
                input_data = np.hstack((data_scaled, data_encoded))
                
                # Predict present price
                prediction = model.predict(input_data)
                predicted_price = round(max(0, prediction[0]), 2)  # Round price to 2 decimal places and ensure non-negative value
                st.success(f"Predicted Present Price: ₹{predicted_price:,.2f} lakhs")
            except ValueError:
                st.error("Please enter valid numerical values where applicable.")

elif page == "About":
    st.title("ℹ️ About this App")
    st.image("cute.png", use_container_width=True)
    st.markdown("## This app predicts the present price of a car based on its past selling price and other parameters using a trained machine learning model.")
