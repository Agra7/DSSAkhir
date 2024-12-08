import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_model():
    with open('DSSAkhir/model_terbaru.pkl', 'rb') as model_file:
        model_dict = pickle.load(model_file)
    return model_dict

def predict(features, model_dict):
    W = model_dict['W']
    b = model_dict['b']

    prediction = np.dot(features, W) + b
    return prediction

def schizophrenia():
    import streamlit as st

    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("Schizophrenia 🤯")
    with col2:
        lang = st.selectbox("Language", ["English", "Indonesia"], label_visibility="collapsed")

    texts = {
        "title": {"English": "Schizophrenia 🤯", "Indonesia": "Skizofrenia 🤯"},
        "name": {"English": "Name", "Indonesia": "Nama"},
        "age": {"English": "Age", "Indonesia": "Usia"},
        "gender": {"English": "Gender", "Indonesia": "Jenis Kelamin"},
        "male": {"English": "Male", "Indonesia": "Pria"},
        "female": {"English": "Female", "Indonesia": "Wanita"},
        "marital_status": {"English": "Marital Status", "Indonesia": "Status Pernikahan"},
        "single": {"English": "Single", "Indonesia": "Belum Menikah"},
        "married": {"English": "Married", "Indonesia": "Menikah"},
        "divorced": {"English": "Divorced", "Indonesia": "Cerai"},
        "widowed": {"English": "Widowed", "Indonesia": "Janda/Duda"},
        "fatigue": {"English": "Fatigue", "Indonesia": "Kelelahan"},
        "slowing": {"English": "Slowing", "Indonesia": "Perlambatan"},
        "pain": {"English": "Pain", "Indonesia": "Rasa Sakit"},
        "hygiene": {"English": "Hygiene", "Indonesia": "Kebersihan"},
        "movement": {"English": "Movement", "Indonesia": "Pergerakan"},
        "submit": {"English": "Submit", "Indonesia": "Kirim"},
        "prediction": {"English": "Prediction", "Indonesia": "Prediksi"},
        "result": {"English": "Prediction Result", "Indonesia": "Hasil Prediksi"},
        "failed": {"English": "Prediction Failed", "Indonesia": "Prediksi Gagal"},
        "low": {"English": "Low Proneness", "Indonesia": "Kecenderungan Rendah"},
        "moderate": {"English": "Moderate Proneness", "Indonesia": "Kecenderungan Sedang"},
        "high": {"English": "High Proneness", "Indonesia": "Kecenderungan Tinggi"},
        "elevated": {"English": "Elevated Proneness", "Indonesia": "Kecenderungan Tinggi"},
        "very_high": {"English": "Very High Proneness", "Indonesia": "Kecenderungan Sangat Tinggi"},
    }

    name = st.text_input(texts["name"][lang])
    age = st.slider(texts["age"][lang], 0, 120, 0)
    gender = st.selectbox(texts["gender"][lang], [texts["male"][lang], texts["female"][lang]])
    marital_status = st.selectbox(
        texts["marital_status"][lang],
        [texts["single"][lang], texts["married"][lang], texts["divorced"][lang], texts["widowed"][lang]],
    )
    fatigue = st.slider(texts["fatigue"][lang], 0, 10, 0)
    slowing = st.slider(texts["slowing"][lang], 0, 10, 0)
    pain = st.slider(texts["pain"][lang], 0, 10, 0)
    hygiene = st.slider(texts["hygiene"][lang], 10, 0, 10)
    movement = st.slider(texts["movement"][lang], 10, 0, 10)

    if st.button(texts["submit"][lang]):
        gender_num = 1 if gender == texts["male"][lang] else 0
        marital_status_map = {
            texts["single"][lang]: 0,
            texts["married"][lang]: 1,
            texts["divorced"][lang]: 2,
            texts["widowed"][lang]: 3,
        }
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

            st.success(f"{texts['prediction'][lang]}: {prediction}")

            def prediction_detail(prediction):
                if prediction < 2:
                    return texts["low"][lang]
                elif 2 <= prediction < 4:
                    return texts["moderate"][lang]
                elif 4 <= prediction < 6:
                    return texts["high"][lang]
                elif 6 <= prediction < 8:
                    return texts["elevated"][lang]
                else:
                    return texts["very_high"][lang]

            detail = prediction_detail(prediction[0])
            st.info(f"{texts['result'][lang]}: {detail}")

        except Exception as e:
            st.error(f"{texts['failed'][lang]}: {e}")

if __name__ == "__main__":
    schizophrenia()
