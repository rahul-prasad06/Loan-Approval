
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle
import streamlit as st
import time as t
# Correct loading of the Random Forest model
path = "D:/DATA SCIENCE CLASS/Bank Loan Approval/loan_prediction.pkl"
rf_model = pickle.load(open(path, "rb"))

def loan_prediction(input_data, rf_model):
    # Convert input data to DataFrame
    data = pd.DataFrame(input_data)
    
    # Calculate total assets
    data["total_assest"] = data["residential_assets_value"] + data["commercial_assets_value"] + data["luxury_assets_value"] + data["bank_asset_value"]
    
    # Calculate monthly income and monthly loan amount
    data["monthly"] = data["income_annum"] / 12
    data["month_loan"] = (data["loan_amount"] / data["loan_term"]) / 12
    
    # Calculate DTI, LTV, and LTI
    data["DTI"] = data["month_loan"] / data["monthly"]  # Monthly loan to income ratio 
    data["LTV"] = data["loan_amount"] / data["total_assest"]  # loan to value ratio
    data['LTI'] = data['loan_amount'] / data['income_annum']   # total loan amount to annual income
    
    # Categorize into bins for LTI, LTV, and DTI
    data['LTI_Category'] = pd.cut(data['LTI'], bins=[0, 3, 5, 100], labels=['Low', 'Moderate', 'High'])
    data['LTV_Category'] = pd.cut(data['LTV'], bins=[0, 0.6, 0.8, 0.95, 100], labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical'])
    data['DTI_Category'] = pd.cut(data['DTI'], bins=[0, 0.2, 0.35, 0.5, 100], labels=['Low', 'Moderate', 'High', 'Critical'])
    
    # Ordinal Encoding
    encoder_lti = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']])
    data["LTI_Category"] = encoder_lti.fit_transform(data[["LTI_Category"]])
    
    encoder_ltv = OrdinalEncoder(categories=[['Low Risk', 'Moderate Risk', 'High Risk', 'Critical']])
    data["LTV_Category"] = encoder_ltv.fit_transform(data[["LTV_Category"]])
    
    encoder_dti = OrdinalEncoder(categories=[['Low', 'Moderate', 'High', 'Critical']])
    data["DTI_Category"] = encoder_dti.fit_transform(data[["DTI_Category"]])
    print(data["LTI_Category"].value_counts())
    print(data["LTI_Category"].unique())
    
    # Scaling the features
    scaler = StandardScaler()
    data[['loan_term', 'cibil_score']] = scaler.fit_transform(data[['loan_term', 'cibil_score']])
    
    # Drop columns that are not needed for prediction
    data.drop(columns=["monthly", "month_loan", 'residential_assets_value', "LTI", "LTV", "DTI",
                       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 'no_of_dependents', 
                       'education', 'self_employed', "income_annum", "loan_amount", "total_assest"], inplace=True)
    
    # Make prediction using the trained random forest model
    prediction = rf_model.predict(data)
    
    return prediction




# Streamlit UI for user input
st.title("RP Finance Service")
st.header("Loan Approval Prediction")
st.subheader("Check Your Eligibility")
st.info("This Model will predict the person's eligibility")

# User Inputs
# Grouping inputs in a form
with st.form("loan_form"):
    Name = st.text_input("Enter Your Name")
    Gender = st.selectbox("Enter Your Gender", ["Male", "Female", "Others"])
    Dependents = st.number_input("Number of Dependents", min_value=0, step=1, format="%d")
    Education = st.selectbox("Education Level", ['Graduate', 'Undergraduate', 'Postgraduate'])
    Self_Employed = st.selectbox("Are you self-employed?", ['Yes', 'No'])
    Income_Annum = st.number_input("Annual Income", min_value=0.0)
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Loan_Term = st.number_input("Loan Term (in years)", min_value=1, step=1, format="%d")
    Cibil_Score = st.number_input("Cibil Score", min_value=300, max_value=900)
    Residential_Assets = st.number_input("Residential Assets Value", min_value=0.1)
    Commercial_Assets = st.number_input("Commercial Assets Value", min_value=0.1)
    Luxury_Assets = st.number_input("Luxury Assets Value", min_value=0.1)
    Bank_Assets = st.number_input("Bank Asset Value", min_value=0.1)
    
    # Button inside the form
    submit_button = st.form_submit_button(label="Predict Eligibility")

if submit_button:
    with st.spinner("Just wait..."):
        t.sleep(2)

    total_assets = Residential_Assets + Commercial_Assets + Luxury_Assets + Bank_Assets
    if Loan_Amount > total_assets:
        st.warning("Loan Amount exceeds total assets. Please revise.")
    else:
        input_data = {
            'education': [Education],
            'self_employed': [Self_Employed],
            'loan_amount': [Loan_Amount],
            'loan_term': [Loan_Term],
            'cibil_score': [Cibil_Score],
            'residential_assets_value': [Residential_Assets],
            'commercial_assets_value': [Commercial_Assets],
            'luxury_assets_value': [Luxury_Assets],
            'bank_asset_value': [Bank_Assets],
            'income_annum': [Income_Annum],
            'no_of_dependents': [Dependents]
        }

        # Call the loan_prediction function
        prediction = loan_prediction(input_data, rf_model)
        if prediction == 1:
            st.success("Loan Approved")
            st.balloons()
        else:
            st.error("Sorry, Loan Not Approved")

st.caption("Thank you for choosing RP Finance Service")
    
