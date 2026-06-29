import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder , StandardScaler 
import pandas as pd
import joblib


## load the train model
model = tf.keras.models.load_model('ann_model.keras')


## load the scaler ,pickle,onehot

scaler = joblib.load('scaler.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
geo_encoder = joblib.load('geo_encoder.pkl')

# streamlit app
st.title("Customer Churan Predication")

# user input
geography = st.selectbox('Geography',geo_encoder.categories_[0])
gender = st.selectbox('Gender',gender_encoder.categories_[0])
age = st.slider('Age',18,92)
st.markdown("""
<style>
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_product = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card ',[0,1])
is_active_member = st.selectbox('Is Active card',[0,1])

# prepare the inut data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_product],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## convert data into one hot encoding
gender_encoded = gender_encoder.transform(
    input_data[['Gender']]
)

gender_df = pd.DataFrame(
    gender_encoded,
    columns=gender_encoder.get_feature_names_out(['Gender'])
)

geo_encoded = geo_encoder.transform(
    input_data[['Geography']]
)

geo_df = pd.DataFrame(
    geo_encoded,
    columns=geo_encoder.get_feature_names_out(['Geography'])
)

# drop categorical columns 
input_data = input_data.drop(
    ['Gender', 'Geography'],
    axis=1
)

# combine everything
input_data = pd.concat(
    [input_data, gender_df, geo_df],
    axis=1
)

# scale input data
input_data_scaled = scaler.transform(input_data)

# predict churn
prediction = model.predict(input_data_scaled)
prediction_probo = prediction[0][0]

st.write(f"Churn Probability: {prediction_probo:.2f}")
if prediction_probo > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")