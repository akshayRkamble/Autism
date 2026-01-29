import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def load_questionnaire_model():
    # Load trained RandomForest and scaler for questionnaire
    import pickle
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'questionnaire_rf.pkl'))
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'questionnaire_scaler.pkl'))
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def load_image_model():
    # Load trained RandomForest and scaler for image
    import pickle
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'image_rf.pkl'))
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'image_scaler.pkl'))
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def extract_image_features_from_upload(img, img_size=(64, 64)):
    img = img.convert('L').resize(img_size)
    arr = np.array(img).flatten()
    return arr.reshape(1, -1)

def main():
    st.title('Autism Spectrum Disorder Prediction')
    st.write('Upload questionnaire answers and a facial image to predict ASD.')

    # Questionnaire input
    st.header('Questionnaire')
    q_cols = [
        'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
        'age','gender','ethnicity','jundice','austim','contry_of_res','used_app_before','result','age_desc','relation'
    ]
    q_input = []
    for col in q_cols:
        val = st.text_input(f"{col}")
        q_input.append(val)

    # Image input
    st.header('Facial Image')
    uploaded_file = st.file_uploader("Choose a facial image", type=["jpg", "jpeg", "png"])

    if st.button('Predict'):
        # Load models
        q_model, q_scaler = load_questionnaire_model()
        i_model, i_scaler = load_image_model()
        # Prepare questionnaire input
        q_input_arr = np.array(q_input).reshape(1, -1)
        q_input_scaled = q_scaler.transform(q_input_arr)
        q_pred_prob = q_model.predict_proba(q_input_scaled)
        # Prepare image input
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img_feat = extract_image_features_from_upload(img)
            img_feat_scaled = i_scaler.transform(img_feat)
            i_pred_prob = i_model.predict_proba(img_feat_scaled)
        else:
            st.error('Please upload a facial image.')
            return
        # Combine predictions
        min_classes = min(q_pred_prob.shape[1], i_pred_prob.shape[1])
        combined_prob = (q_pred_prob[0][:min_classes] + i_pred_prob[0][:min_classes]) / 2
        pred = np.argmax(combined_prob)
        st.success(f'Predicted ASD class: {pred}')

if __name__ == '__main__':
    main()
