
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("columns.json") as f:
    model_columns = json.load(f)

users = {"admin": "1234"}

if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in users and users[user] == pwd:
            st.session_state.login = True
        else:
            st.error("Invalid credentials")

def app():
    st.title("🎓 Placement Prediction System")

    menu = st.sidebar.selectbox("Menu", ["Single", "CSV", "Dashboard"])

    if menu == "Single":
        gender = st.selectbox("Gender", ["M","F"])
        ssc_p = st.slider("SSC %", 0.0, 100.0)
        hsc_p = st.slider("HSC %", 0.0, 100.0)
        degree_p = st.slider("Degree %", 0.0, 100.0)
        workex = st.selectbox("Work Experience", ["Yes","No"])
        etest_p = st.slider("E-test %", 0.0, 100.0)
        mba_p = st.slider("MBA %", 0.0, 100.0)

        data = {
            "ssc_p": ssc_p,
            "hsc_p": hsc_p,
            "degree_p": degree_p,
            "etest_p": etest_p,
            "mba_p": mba_p,
            "gender_M": 1 if gender=="M" else 0,
            "workex_Yes": 1 if workex=="Yes" else 0
        }

        df = pd.DataFrame([data])
        df = df.reindex(columns=model_columns, fill_value=0)

        scaled = scaler.transform(df)

        if st.button("Predict"):
            res = model.predict(scaled)
            if res[0] == 1:
                st.success("Placed")
            else:
                st.error("Not Placed")

    elif menu == "CSV":
        file = st.file_uploader("Upload CSV")
        if file:
            df = pd.read_csv(file)
            df = pd.get_dummies(df)
            df = df.reindex(columns=model_columns, fill_value=0)

            scaled = scaler.transform(df)
            pred = model.predict(scaled)

            df["Prediction"] = pred
            st.write(df)

    elif menu == "Dashboard":
        df = pd.read_csv("Placement_Data_Full_Class.csv")
        st.bar_chart(df["status"].value_counts())
        st.line_chart(df["ssc_p"])

if st.session_state.login:
    app()
else:
    login()
