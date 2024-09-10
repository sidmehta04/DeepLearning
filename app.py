import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('Label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoding_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Custom CSS for better visuals
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            color: #333333;
        }
        .main {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
        }
        h1 {
            font-family: 'Arial', sans-serif;
            color: #0066cc;
        }
        label {
            font-weight: bold;
            color: #0066cc;
        }
        .stButton button {
            background-color: #0066cc;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            border: none;
        }
        .stButton button:hover {
            background-color: #004d99;
            color: #e6e6e6;
        }
        .prediction-probability {
            font-size: 24px;
            color: #ff6600;
        }
        .churn-prediction {
            font-weight: bold;
            font-size: 20px;
            color: #cc0000;
        }
    </style>
""", unsafe_allow_html=True)

# App title with enhanced styling
st.title('Customer Churn Prediction')

# Organize inputs using columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)

with col2:
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the result with enhanced styling
st.markdown(f'<div class="prediction-probability">Churn Probability: {prediction_proba:.2f}</div>', unsafe_allow_html=True)

if prediction_proba > 0.5:
    st.markdown('<div class="churn-prediction">The customer is likely to churn.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="churn-prediction" style="color: #009933;">The customer is not likely to churn.</div>', unsafe_allow_html=True)
