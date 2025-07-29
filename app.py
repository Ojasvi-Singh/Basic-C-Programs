import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Prevent TkAgg backend crash

# Load model & feature list
with open(r'C:\Users\singh\college assistent\medical\model.pkl', 'rb') as f:
    model, feature_names = pickle.load(f)

# UI config
st.set_page_config(page_title="Hospital Readmission Predictor", layout="centered")
st.title("🏥 AI Readmission Risk Predictor")
st.markdown("Predict hospital readmission risk and explain it with SHAP 🔬")

# Input dropdowns
race_options = ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other']
gender_options = ['Male', 'Female']
insulin_options = ['No', 'Up', 'Down', 'Steady']

# User input
user_input = {
    'race': st.selectbox("Race", race_options),
    'gender': st.radio("Gender", gender_options),
    'age': st.slider("Age", 0, 100, 50),
    'time_in_hospital': st.slider("Time in Hospital", 1, 14, 3),
    'num_lab_procedures': st.slider("Lab Procedures", 0, 132, 40),
    'num_procedures': st.slider("Procedures", 0, 6, 1),
    'num_medications': st.slider("Medications", 1, 81, 10),
    'number_outpatient': st.slider("Outpatient Visits", 0, 42, 0),
    'number_emergency': st.slider("Emergency Visits", 0, 76, 0),
    'number_inpatient': st.slider("Inpatient Visits", 0, 21, 0),
    'number_diagnoses': st.slider("Number of Diagnoses", 1, 16, 5),
    'insulin': st.selectbox("Insulin Level", insulin_options)
}

# Encoders
label_encoders = {
    'race': {v: i for i, v in enumerate(race_options)},
    'gender': {'Female': 0, 'Male': 1},
    'insulin': {'No': 0, 'Up': 1, 'Down': 2, 'Steady': 3}
}

if st.button("Predict Outcome"):
    try:
        # Process input
        input_df = pd.DataFrame([user_input])
        for col in label_encoders:
            input_df[col] = input_df[col].map(label_encoders[col])
        input_df = input_df[feature_names]

        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Class labels
        labels = {
            0: "No Readmission",
            1: "Readmission After 30 Days",
            2: "Readmission Within 30 Days"
        }

        st.success(f"✅ Predicted Outcome: **{labels.get(prediction, 'Unknown')}**")

        # Show probabilities
        st.markdown("### 🔬 Prediction Probabilities")
        for i, p in enumerate(proba):
            st.write(f"{['🟢', '🟠', '🔴'][i]} {labels[i]}: {p*100:.2f}%")

        # SHAP explanation
        st.markdown("### 🧠 Why This Prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Use predicted class safely
        safe_index = prediction if prediction < len(shap_values) else 0
        shap_row = shap_values[safe_index][0]

        # Top 5 feature impacts
        contributions = sorted(
            zip(feature_names, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        for feat, val in contributions:
            arrow = "↑" if val > 0 else "↓"
            st.write(f"{arrow} `{feat}`: {val:.3f}")

        # ✅ SHAP Bar Chart (NO MISMATCH!)
        st.markdown("#### 📊 Feature Impact (Bar Chart)")
        fig, ax = plt.subplots()

        # Re-encode user_input cleanly
        input_encoded_df = pd.DataFrame([user_input])
        for col in label_encoders:
            input_encoded_df[col] = input_encoded_df[col].map(label_encoders[col])
        input_encoded_df = input_encoded_df[feature_names]

        shap_df = pd.DataFrame([shap_row], columns=feature_names)
        shap.summary_plot(shap_df.values, input_encoded_df, plot_type="bar", show=False)
        st.pyplot(fig)

        # Treatment Suggestions
        if prediction == 2:
            st.warning("⚠️ High Risk (<30 days)")
            st.markdown("""
            ### 🔴 Treatment Plan:
            - 💉 Intensify insulin  
            - 🧪 Order A1C & glucose tests  
            - 📅 Follow-up in 7 days  
            - 📘 Educate patient on compliance
            """)
        elif prediction == 1:
            st.info("🟠 Moderate Risk (>30 days)")
            st.markdown("""
            ### 🟠 Treatment Plan:
            - 📋 Ensure medication adherence  
            - 🍎 Recommend diet/lifestyle changes  
            - 📅 Follow-up in 3–4 weeks
            """)
        else:
            st.success("🟢 Low Risk")
            st.markdown("""
            ### 🟢 Maintenance Plan:
            - ✅ Maintain current treatment  
            - 📅 Routine checkup in 2–3 months
            """)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
