

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ------------------ LOAD MODEL, SCALER, PCA ------------------
model = pickle.load(open("credit_risk_app.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

st.title("German Credit Risk Prediction App")
st.write("Enter customer details to predict Good/Bad Credit Risk.")

# ------------------ USER INPUT FIELDS ------------------

ID = st.number_input("Customer ID", min_value=1, value=1)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)

Sex = st.selectbox("Sex", ["male", "female"])
Housing = st.selectbox("Housing", ["own", "rent", "free"])
Saving = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich", "Unknown"])
Checking = st.selectbox("Checking account", ["little", "moderate", "rich", "Unknown"])

Job = st.selectbox("Job", [0, 1, 2, 3])
CreditAmount = st.number_input("Credit Amount", min_value=0, value=1000)
Duration = st.number_input("Duration in Months", min_value=1, value=12)

Purpose = st.selectbox(
    "Purpose",
    ["business", "car", "domestic appliances", "education",
     "furniture/equipment", "radio/TV", "repairs", "vacation/others"]
)

# ------------------ LABEL ENCODING (Same as Training) ------------------
label_maps = {
    "Sex": {"male": 1, "female": 0},
    "Housing": {"own": 2, "rent": 1, "free": 0},
    "Saving accounts": {"little": 0, "moderate": 1, "rich": 2, "quite rich": 3, "Unknown": 4},
    "Checking account": {"little": 0, "moderate": 1, "rich": 2, "Unknown": 3}
}

Sex = label_maps["Sex"][Sex]
Housing = label_maps["Housing"][Housing]
Saving = label_maps["Saving accounts"][Saving]
Checking = label_maps["Checking account"][Checking]

# ------------------ ONE-HOT ENCODING FOR PURPOSE ------------------
purpose_values = {
    "business": 0,
    "car": 0,
    "domestic appliances": 0,
    "education": 0,
    "furniture/equipment": 0,
    "radio/TV": 0,
    "repairs": 0,
    "vacation/others": 0
}
purpose_values[Purpose] = 1

# ------------------ BUILD FINAL INPUT ------------------
# *** Ensure exact order of 17 training features ***

input_data = pd.DataFrame([{
    "Id": ID,
    "Age": Age,
    "Sex": Sex,
    "Job": Job,
    "Housing": Housing,
    "Saving accounts": Saving,
    "Checking account": Checking,
    "Credit amount": CreditAmount,
    "Duration": Duration,
    "business": purpose_values["business"],
    "car": purpose_values["car"],
    "domestic appliances": purpose_values["domestic appliances"],
    "education": purpose_values["education"],
    "furniture/equipment": purpose_values["furniture/equipment"],
    "radio/TV": purpose_values["radio/TV"],
    "repairs": purpose_values["repairs"],
    "vacation/others": purpose_values["vacation/others"]
}])

# ------------------ APPLY SCALER + PCA ------------------
scaled = scaler.transform(input_data)
pca_features = pca.transform(scaled)

# ------------------ PREDICTION ------------------
if st.button("Predict"):
    prediction = model.predict(pca_features)[0]

    if prediction == 1:
        st.success("✔ GOOD Customer (Low Risk)")
    else:
        st.error("❌ BAD Customer (High Risk)")
