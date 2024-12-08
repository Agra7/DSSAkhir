import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pickle

def load_model():
    with open('DSSAkhir/model_terbaru.pkl', 'rb') as model_file:
        model_dict = pickle.load(model_file)
        print("Loaded dictionary keys:", model_dict.keys())
    return model_dict

def predict(features, model_dict):
    W = model_dict['W']
    b = model_dict['b']

    prediction = np.dot(features, W) + b
    return prediction

def schizophrenia():
    import streamlit as st

    st.title("Schizophrenia 🤯")

    name = st.text_input("Name")
    age = st.slider("Age", 0, 120, 0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    fatigue = st.slider("Fatigue", 0, 10, 0)
    slowing = st.slider("Slowing", 0, 10, 0)
    pain = st.slider("Pain", 0, 10, 0)
    hygiene = st.slider("Hygiene", 10, 0, 10)
    movement = st.slider("Movement", 10, 0, 10)

    if st.button("Submit"):
        gender_num = 1 if gender == "Male" else 0
        marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
        marital_status_num = marital_status_map[marital_status]

        # Placeholder
        null_1 = 0
        null_2 = 0
        null_3 = 0

        features = np.array([[age, gender_num, marital_status_num, fatigue, slowing, pain, hygiene, movement, 
                            null_1, null_2, null_3]])

        model_dict = load_model()
        try:
            prediction = predict(features, model_dict)

            st.success(f"Prediction: {prediction}")

            def prediction_detail(prediction):
                if prediction < 2:
                    return "Low Proneness"
                elif 2 <= prediction < 4:
                    return "Moderate Proneness"
                elif 4 <= prediction < 6:
                    return "High Proneness"
                elif 6 <= prediction < 8:
                    return "Elevated Proneness"
                else:
                    return "Very High Proneness"

            detail = prediction_detail(prediction[0])
            st.info(f"Interpreted Result: {detail}")

        except Exception as e:
            st.error(f"Prediction Failed: {e}")

if __name__ == "__main__":
    schizophrenia()
