import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Placement Predictor", layout="centered")

# ---------------- LOAD FILES ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("columns.json") as f:
    model_columns = json.load(f)

# ---------------- LOGIN ----------------
users = {"admin": "1234"}

if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Login System")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in users and users[user] == pwd:
            st.session_state.login = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

# ---------------- MAIN APP ----------------
def app():
    st.title("🎓 AI-Based Placement Prediction System")

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.login = False

    menu = st.sidebar.selectbox(
        "Menu",
        ["Single Prediction", "Batch CSV", "Dashboard"]
    )

    # ---------------- SINGLE ----------------
    if menu == "Single Prediction":
        st.subheader("Enter Student Details")

        gender = st.selectbox("Gender", ["M", "F"])
        ssc_p = st.slider("SSC %", 0.0, 100.0)
        hsc_p = st.slider("HSC %", 0.0, 100.0)
        degree_p = st.slider("Degree %", 0.0, 100.0)
        workex = st.selectbox("Work Experience", ["Yes", "No"])
        etest_p = st.slider("E-test %", 0.0, 100.0)
        mba_p = st.slider("MBA %", 0.0, 100.0)

        # Input
        data = {
            "ssc_p": ssc_p,
            "hsc_p": hsc_p,
            "degree_p": degree_p,
            "etest_p": etest_p,
            "mba_p": mba_p,
            "gender_M": 1 if gender == "M" else 0,
            "workex_Yes": 1 if workex == "Yes" else 0
        }

        df = pd.DataFrame([data])
        df = df.reindex(columns=model_columns, fill_value=0)

        scaled = scaler.transform(df)

        if st.button("Predict"):
            result = model.predict(scaled)
            prob = model.predict_proba(scaled)[0][1]

            if result[0] == 1:
                st.success(f"🎉 Placed (Confidence: {prob*100:.2f}%)")
            else:
                st.error(f"❌ Not Placed (Confidence: {(1-prob)*100:.2f}%)")

    # ---------------- CSV ----------------
    elif menu == "Batch CSV":
        st.subheader("Upload CSV for Prediction")

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)

            df_encoded = pd.get_dummies(df)
            df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

            scaled = scaler.transform(df_encoded)

            pred = model.predict(scaled)
            prob = model.predict_proba(scaled)[:, 1]

            df["Prediction"] = pred
            df["Confidence"] = prob

            st.write("Prediction Results", df)

            # Summary
            st.subheader("Summary")
            st.write("Total Students:", len(df))
            st.write("Placed:", (df["Prediction"] == 1).sum())
            st.write("Not Placed:", (df["Prediction"] == 0).sum())

            # Download
            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "placement_results.csv"
            )

    # ---------------- DASHBOARD ----------------
    elif menu == "Dashboard":
        st.subheader("📊 Data Dashboard")

        df = pd.read_csv("Placement_Data_Full_Class.csv")

        # Placement chart
        st.write("Placement Distribution")
        st.bar_chart(df["status"].map({'Placed': 1, 'Not Placed': 0}).value_counts())

        # SSC graph
        st.write("SSC Percentage Trend")
        st.line_chart(df["ssc_p"])

        # Insights
        st.subheader("Insights")

        placed_avg = df[df["status"] == "Placed"]["ssc_p"].mean()
        not_placed_avg = df[df["status"] == "Not Placed"]["ssc_p"].mean()

        st.write(f"Average SSC (Placed): {placed_avg:.2f}")
        st.write(f"Average SSC (Not Placed): {not_placed_avg:.2f}")

# ---------------- RUN ----------------
if st.session_state.login:
    app()
else:
    login()
