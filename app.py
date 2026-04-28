import streamlit as st
import pandas as pd
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
users = {"Rajiv": "2003"}

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

    if st.sidebar.button("Logout"):
        st.session_state.login = False

    menu = st.sidebar.selectbox(
        "Menu",
        ["Single Prediction", "Batch CSV", "Dashboard"]
    )

    # ---------------- SINGLE PREDICTION ----------------
    if menu == "Single Prediction":

        gender = st.selectbox("Gender", ["M", "F"])
        ssc_p = st.slider("SSC %", 0.0, 100.0)
        hsc_p = st.slider("HSC %", 0.0, 100.0)
        degree_p = st.slider("Degree %", 0.0, 100.0)
        workex = st.selectbox("Work Experience", ["Yes", "No"])
        etest_p = st.slider("E-test %", 0.0, 100.0)
        mba_p = st.slider("MBA %", 0.0, 100.0)

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

            # Result
            if result[0] == 1:
                st.success(f"🎉 Placed (Confidence: {prob*100:.2f}%)")
            else:
                st.error(f"❌ Not Placed (Confidence: {(1-prob)*100:.2f}%)")

            # ---------------- PROBABILITY BAR ----------------
            st.progress(int(prob * 100))
            st.write(f"Placement Probability: {prob*100:.2f}%")

            # ---------------- REQUIREMENTS CHECK ----------------
            st.subheader("📌 Placement Requirements Analysis")

            suggestions = []

            if ssc_p < 60:
                suggestions.append("Improve SSC > 60%")
            if hsc_p < 60:
                suggestions.append("Improve HSC > 60%")
            if degree_p < 65:
                suggestions.append("Increase Degree > 65%")
            if etest_p < 70:
                suggestions.append("Improve Aptitude > 70%")
            if mba_p < 65:
                suggestions.append("Improve MBA > 65%")
            if workex == "No":
                suggestions.append("Gain internship/work experience")

            if len(suggestions) == 0:
                st.success("✅ You meet most placement requirements!")
            else:
                for s in suggestions:
                    st.warning(s)

            # ---------------- PERFORMANCE SCORE ----------------
            st.subheader("📈 Student Performance Score")

            score = (ssc_p + hsc_p + degree_p + etest_p + mba_p) / 5
            st.metric("Overall Score", f"{score:.2f}/100")

            if score > 75:
                st.success("Excellent Profile 🔥")
            elif score > 60:
                st.info("Good Profile 👍")
            else:
                st.warning("Needs Improvement ⚠️")

            # ---------------- SKILLS ----------------
            st.subheader("💡 Skills Required")

            skills = [
                "Communication Skills",
                "Aptitude & Logical Reasoning",
                "Programming / Technical Skills",
                "Problem Solving",
                "Teamwork & Confidence",
                "Internship Experience"
            ]

            for skill in skills:
                st.write("✔", skill)

    # ---------------- CSV ----------------
    elif menu == "Batch CSV":

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

            st.write(df)

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "placement_results.csv"
            )

    # ---------------- DASHBOARD ----------------
    elif menu == "Dashboard":

        df = pd.read_csv("Placement_Data_Full_Class.csv")

        st.subheader("Placement Distribution")
        st.bar_chart(df["status"].map({'Placed':1,'Not Placed':0}).value_counts())

        st.subheader("SSC Trend")
        st.line_chart(df["ssc_p"])

# ---------------- RUN ----------------
if st.session_state.login:
    app()
else:
    login()
