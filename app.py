import streamlit as st
import pandas as pd
import numpy as np
from modeling_cat import train_model, predict_score, pipeline, features, categorical_features


st.markdown("""
    <h2 style='text-align: center;'>
        <a href='https://docs.google.com/document/d/1Ur949GM5mvjZjpTAjn-Cnyd4nKmzX1He6tdusI5oNS0/edit?usp=sharing' target='_blank' rel='noopener noreferrer'>View our whitepaper</a>
    </h2>
    """, unsafe_allow_html=True)


st.title('COMPAS Score Prediction App')

st.write("""
This app predicts the COMPAS score based on input features.
Please fill in the following information:
""")

# Create input fields for features
age = st.number_input('Age', min_value=0, max_value=100, value=30)
juv_fel_count = st.number_input('Juvenile Felony Count', min_value=0, max_value=20, value=0)
juv_misd_count = st.number_input('Juvenile Misdemeanor Count', min_value=0, max_value=20, value=0)
juv_other_count = st.number_input('Juvenile Other Count', min_value=0, max_value=20, value=0)
priors_count = st.number_input('Priors Count', min_value=0, max_value=50, value=0)
days_b_screening_arrest = st.number_input('Days Between Screening and Arrest', min_value=-500, max_value=500, value=0)

# Create input fields for categorical features
sex = st.selectbox('Sex', ['Female', 'Male'])
race = st.selectbox('Race', ['African-American', 'Caucasian', 'Hispanic', 'Native American', 'Other'])
c_charge_degree = st.selectbox('Charge Degree', ['F', 'M'])

if st.button('Predict COMPAS Score'):
    # Train the model before launching the app
    # train_model()
    # Prepare input data
    input_data = {
        'age': age,
        'juv_fel_count': juv_fel_count,
        'juv_misd_count': juv_misd_count,
        'juv_other_count': juv_other_count,
        'priors_count': priors_count,
        'days_b_screening_arrest': days_b_screening_arrest,
        'sex': sex,
        'race': race,
        'c_charge_degree': c_charge_degree
    }

    # Make prediction
    prediction = predict_score(input_data)

    st.write(f'The predicted COMPAS score is: **{prediction}**')

    # Display feature importances
    st.write('### Feature Importances')
    feature_importance = pipeline.named_steps['model'].feature_importances_
    feature_names = (pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())
    feature_names = features + feature_names
    
    importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importances = importances.sort_values('importance', ascending=False)
    
    st.bar_chart(importances.set_index('feature')['importance'])
