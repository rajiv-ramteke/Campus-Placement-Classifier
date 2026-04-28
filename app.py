import streamlit as st
import pandas as pd
import joblib
import json

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Placement Predictor", layout="centered")

# ---------------- DARK UI ----------------
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.stButton>button {background-color: #4CAF50; color: white;}
</style>
""", unsafe_allow_html=True)

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
    st.title("🔐 Login")
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

    st.markdown("## 🚀 Smart Placement Advisor")

    if st.sidebar.button("Logout"):
        st.session_state.login = False

    menu = st.sidebar.selectbox(
        "Menu",
        ["Single Prediction", "Batch CSV", "Dashboard", "What If Simulator", "Resume Analyzer"]
    )

    # ---------------- SINGLE ----------------
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

            if result[0] == 1:
                st.success(f"🎉 Placed (Confidence: {prob*100:.2f}%)")
            else:
                st.error(f"❌ Not Placed (Confidence: {(1-prob)*100:.2f}%)")

            st.progress(int(prob * 100))

            # Suggestions
            st.subheader("📌 Suggestions")
            if degree_p < 65:
                st.warning("Improve Degree %")
            if etest_p < 70:
                st.warning("Improve Aptitude")
            if workex == "No":
                st.warning("Get Internship")

            # Score
            score = (ssc_p + hsc_p + degree_p + etest_p + mba_p) / 5
            st.metric("Performance Score", f"{score:.2f}/100")

            # Company Recommendation
            st.subheader("🏢 Recommended Companies")
            if degree_p > 75:
                st.write("Google, Microsoft, Amazon")
            elif degree_p > 65:
                st.write("TCS, Infosys, Wipro")
            else:
                st.write("Startups & Internships")

    # ---------------- CSV ----------------
    elif menu == "Batch CSV":

        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)

            df_encoded = pd.get_dummies(df)
            df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

            scaled = scaler.transform(df_encoded)

            df["Prediction"] = model.predict(scaled)
            df["Confidence"] = model.predict_proba(scaled)[:, 1]

            st.write(df)

            st.download_button(
                "Download",
                df.to_csv(index=False),
                "results.csv"
            )

    # ---------------- DASHBOARD ----------------
    elif menu == "Dashboard":

        df = pd.read_csv("Placement_Data_Full_Class.csv")

        st.subheader("Placement Distribution")
        st.bar_chart(df["status"].map({'Placed':1,'Not Placed':0}).value_counts())

        st.subheader("SSC Trend")
        st.line_chart(df["ssc_p"])

        # Feature Importance (fixed)
        st.subheader("Top Features")
        try:
            importance = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": model_columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(feat_df.set_index("Feature"))
        except:
            st.warning("Feature importance not available")

    # ---------------- WHAT IF ----------------
    elif menu == "What If Simulator":

        st.subheader("What If Analysis")

        ssc = st.slider("SSC", 0.0, 100.0, 60.0)
        hsc = st.slider("HSC", 0.0, 100.0, 60.0)
        degree = st.slider("Degree", 0.0, 100.0, 65.0)
        etest = st.slider("E-test", 0.0, 100.0, 70.0)
        mba = st.slider("MBA", 0.0, 100.0, 65.0)

        sim = pd.DataFrame([{
            "ssc_p": ssc,
            "hsc_p": hsc,
            "degree_p": degree,
            "etest_p": etest,
            "mba_p": mba,
            "gender_M": 1,
            "workex_Yes": 1
        }])

        sim = sim.reindex(columns=model_columns, fill_value=0)
        sim_scaled = scaler.transform(sim)

        prob = model.predict_proba(sim_scaled)[0][1]

        st.progress(int(prob * 100))
        st.write(f"Chance: {prob*100:.2f}%")

    # ---------------- RESUME ----------------
    elif menu == "Resume Analyzer":

        st.subheader("Resume Scoring")

        text = st.text_area("Paste Resume")

        if st.button("Analyze"):

            score = 0
            keywords = ["python","ml","project","internship","sql","teamwork"]

            for k in keywords:
                if k in text.lower():
                    score += 10

            st.metric("Score", f"{score}/100")

# ---------------- RUN ----------------
if st.session_state.login:
    app()
else:
    login()
