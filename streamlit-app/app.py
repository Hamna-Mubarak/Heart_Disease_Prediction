"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
Developer: Hamna Mubarak (FA24-BSE-046, Section A)

Loads the trained Support Vector Classifier (heart_disease_model.pkl) produced
by Heart_Disease_Prediction.ipynb, and recreates the exact same LabelEncoders
(fit on the same fixed category lists) used during training, so predictions
match the notebook exactly.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced, in the repo root)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent.parent / "heart_disease_model.pkl"

SEX_VALUES = ["Male", "Female"]
CHEST_PAIN_VALUES = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
MAJOR_VESSELS_VALUES = ["Zero", "One", "Two", "Three"]
THAL_VALUES = ["Normal", "Fixed Defect", "Reversible Defect"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in Step 5.1 of the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes.
    """
    return {
        "Sex": LabelEncoder().fit(SEX_VALUES),
        "ChestPainType": LabelEncoder().fit(CHEST_PAIN_VALUES),
        "MajorVessels": LabelEncoder().fit(MAJOR_VESSELS_VALUES),
        "Thal": LabelEncoder().fit(THAL_VALUES),
    }


model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Built from the **UCI Cleveland Heart Disease** dataset. Select a patient's details
    below and the trained Support Vector Classifier (SVC) will predict whether the
    patient has heart disease.

    *Model accuracy on held-out test data: **88%** (10-fold cross validation: 80%).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", SEX_VALUES)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN_VALUES, index=3)
with col2:
    major_vessels = st.selectbox("Major Vessels Coloured by Fluoroscopy", MAJOR_VESSELS_VALUES)
    thal = st.selectbox("Thal (Thallium Stress Test)", THAL_VALUES)

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "MajorVessels": [major_vessels],
        "Thal": [thal],
    })

    encoded = pd.DataFrame({
        column: encoders[column].transform(user_input[column]) for column in user_input.columns
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE: YES")
    else:
        st.success("### ✅ HEART DISEASE: NO")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Developed by Hamna Mubarak (FA24-BSE-046, Section A) for Machine Learning Fundamentals (AIC354), "
    "COMSATS University Islamabad, Lahore Campus. For educational purposes only - not medical advice."
)
