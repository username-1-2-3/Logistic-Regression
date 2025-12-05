import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction App (Fixed input shape)")

st.markdown("Enter patient details (all fields required).")

# --- INPUT FIELDS (make sure you include the EXACT features used in training) ---
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70, step=1)
skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20, step=1)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
dpf = st.number_input("DiabetesPedigreeFunction (DPF)", min_value=0.0, max_value=5.0, value=0.5, format="%.4f")
age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)

if st.button("Predict"):
    # Build input in the same order used in training
    user_data = np.array([pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi,
                          dpf, age], dtype=float)

    # Make sure shape is (1, n_features)
    user_data = user_data.reshape(1, -1)
    st.write("Input shape:", user_data.shape)

    # Optional: show what model expects
    st.write("Model expects n_features_in_ =", getattr(model, "n_features_in_", "unknown"))

    # If scaler was used during training, transform the data
    try:
        scaled_data = scaler.transform(user_data)
    except Exception as e:
        st.error("Error when transforming input with scaler: " + str(e))
        st.stop()

    # Now predict
    try:
        pred = model.predict(scaled_data)
        prob = model.predict_proba(scaled_data)[0,1]
    except Exception as e:
        st.error("Prediction error: " + str(e))
        st.stop()

    if pred[0] == 1:
        st.error(f"Prediction: High chance of Diabetes (probability = {prob:.2f})")
    else:
        st.success(f"Prediction: Low chance of Diabetes (probability = {prob:.2f})")
