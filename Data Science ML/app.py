import streamlit as st
import pickle
import numpy as np

# Load Model
with open("resume_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("📄 Resume Shortlisting System")

# Encode education
education_dict = {
    "High School": 0,
    "Bachelors": 1,
    "Masters": 2,
    "PhD": 3
}

education_label = st.selectbox(
    "Education Level",
    options=list(education_dict.keys())
)

education_level = education_dict[education_label]

years_experience = st.number_input("Years of Experience", min_value=0.0)
skills_match_score = st.number_input("Skills Match Score", min_value=0.0)

if st.button("Check Resume"):

    input_data = np.array([[years_experience,
                            skills_match_score,
                            education_level]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Resume Shortlisted")
    else:
        st.error("❌ Resume Not Shortlisted")

