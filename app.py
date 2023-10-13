import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load the pre-trained XGBoost model
xgb_model = joblib.load("xgb_model.pkl")

columns = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count',
           'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 
           'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 
           'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'bank_months_count', 
           'has_other_cards', 'proposed_credit_limit', 'foreign_request', 'session_length_in_minutes', 
           'keep_alive_session', 'device_distinct_emails_8w', 'device_fraud_count', 'month', 'payment_type_AA', 
           'payment_type_AB', 'payment_type_AC', 'payment_type_AD', 'payment_type_AE', 'employment_status_CA', 
           'employment_status_CB', 'employment_status_CC', 'employment_status_CD', 'employment_status_CE', 
           'employment_status_CF', 'employment_status_CG', 'housing_status_BA', 'housing_status_BB', 
           'housing_status_BC', 'housing_status_BD', 'housing_status_BE', 'housing_status_BF', 'housing_status_BG', 
           'source_INTERNET', 'source_TELEAPP', 'device_os_linux', 'device_os_macintosh', 'device_os_other', 
           'device_os_windows', 'device_os_x11']

# Create a Streamlit app
st.title("Bank Account Fraud Detection")

st.write("Aplikasi ini memprediksi kemungkinan penipuan rekening bank.")

# Create input fields for each feature
income = st.number_input("Income", min_value=0.0, value=0.0)
name_email_similarity = st.number_input("Name-Email Similarity", min_value=0.0, max_value=1.0, value=0.0)
prev_address_months_count = st.number_input("Previous Address Months Count", min_value=0.0, value=0.0)
current_address_months_count = st.number_input("Current Address Months Count", min_value=0.0, value=0.0)

# Create a dictionary with user input
user_input = {
    'income': income,
    'name_email_similarity': name_email_similarity,
    'prev_address_months_count': prev_address_months_count,
    'current_address_months_count': current_address_months_count
}

# Create the feature_vector DataFrame
feature_vector = pd.DataFrame([user_input], columns=columns)

# Create a button to make predictions
if st.button("Predict Fraud"):
    # Predict using the pre-trained XGBoost model
    fraud_prob = xgb_model.predict_proba(feature_vector)

    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"Kemungkinan penipuan adalah : {fraud_prob[0][1]:.2%}")

st.write("Disclaimer: This is a simplified demonstration of a fraud detection model.")
st.write("By. Kelompok 4_Candra")