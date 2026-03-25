import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("Customer Churn Prediction System")
st.markdown("AI-powered churn prediction with explainability")

# Sidebar inputs
st.sidebar.header("Input Parameters")

tenure = st.sidebar.slider("Tenure", 0, 60, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 200, 70)
total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Features
contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

st.write("### Customer Input Summary")
st.write({
    "Tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment
})

# Prediction
if st.button("Predict Churn"):
    with st.spinner("Analyzing..."):

        data = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        result = response.json()

        proba = result["probability"]

        st.subheader("Prediction Result")

        # Use threshold 
        if proba > 0.65:
            st.error(f"Customer is likely to CHURN ({proba:.2f})")
        else:
            st.success(f"Customer will STAY ({proba:.2f})")

        # SHAP Table
        st.subheader("Feature Contribution")

        shap_df = pd.DataFrame({
            "Feature": result["feature_names"],
            "Impact": result["shap_values"]
        })

        shap_df["abs_impact"] = shap_df["Impact"].abs()
        shap_df = shap_df.sort_values(by="abs_impact", ascending=False).head(10)

        st.dataframe(shap_df[["Feature", "Impact"]])

        # Plot
        st.subheader("Top Feature Impacts")

        plt.figure()
        plt.barh(shap_df["Feature"], shap_df["Impact"])
        plt.gca().invert_yaxis()
        plt.xlabel("Impact on Prediction")
        plt.title("Feature Contribution")

        st.pyplot(plt)