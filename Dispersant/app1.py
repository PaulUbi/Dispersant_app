import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from playsound import playsound
import pickle
import os
import sklearn

# Define the parameters
params = {
    'objective': 'reg:squarederror',
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'colsample_bytree': 1.0,
    'gamma': 0.1,
    'grow_policy': 'depthwise',
    'learning_rate': 0.1,
    'max_bin': 256,
    'max_cat_to_onehot': 4,
    'max_depth': 3,
    'min_child_weight': 1,
    'missing': float('nan'),
    'n_estimators': 200,
    'n_jobs': 0,
    'num_parallel_tree': 1,
    'random_state': 42,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'sampling_method': 'uniform',
    'subsample': 0.5,
    'tree_method': 'exact',
    'validate_parameters': 1
}

# Initialize the XGBoost model
XGBoost_final = xgb.XGBRegressor(**params)




# # Load the saved XGBoost model
# model_file_path = "xgboost_model.model"  # Specify the path and filename of your saved model
# XGBoost_final = xgb.Booster()
# XGBoost_final.load_model(model_file_path)

# Define the app interface

st.title('Dispersant Recommender System')

# Create two columns for the images
col1, col2 = st.columns(2)

# Add the dispersant image to the first column
col1.image('Dispersant_app/Dispersant/Dipersant_image.png', caption='Oil Dispersant', width=300)

# Add your school's logo image to the second column
col2.image('KNRTU.jpg', caption='School Logo', width=200)


#####
st.markdown('''
This app predicts the efficiency of oil dispersants based on input parameters.
''')

# Input fields for user input
st.sidebar.header('Input Parameters')
st.sidebar.markdown('Adjust the parameters below:')
temperature = st.sidebar.number_input('Temperature (C)', min_value=0.0, max_value=30.0, value=25.0)
salinity = st.sidebar.number_input('Salinity (g/L)', min_value=0.0, max_value=35.0, value=15.0)
viscosity = st.sidebar.number_input('Kinematic Viscosity (mm^2/c)', min_value=0.0, max_value=80.0, value=50.0)
density = st.sidebar.number_input('Density', min_value=0.5, max_value=0.9, value=0.7)
oil_field = st.sidebar.selectbox('Oil Field', ["Хохряковское", "Усинское", "Правдинское", "Нагорн.(Турней)", "Нагорн.(Башкир)"])
dispersant_ratio = st.sidebar.selectbox('Dispersant Ratio', ["1:10", "1:20"])

# Map the selected oil field to its corresponding encoded value
oil_field_encoded = {"Хохряковское": 4, "Усинское": 3, "Правдинское": 2, "Нагорн.(Турней)": 1, "Нагорн.(Башкир)": 0}
encoded_oil_field = oil_field_encoded[oil_field]

# Map the selected dispersant ratio to its corresponding encoded value
dispersant_ratio_encoded = {"1:10": 1, "1:20": 0}
encoded_dispersant_ratio = dispersant_ratio_encoded[dispersant_ratio]

# Button to trigger predictions
if st.sidebar.button('Predict'):
    # Prepare input data
    input_data = np.array([[temperature, salinity, viscosity, density, encoded_oil_field, encoded_dispersant_ratio]])  # Add other input values here
    
    # Make prediction
    prediction = XGBoost_final.predict(xgb.DMatrix(input_data))
    
    # Display prediction
    st.subheader('Prediction Result')
    if prediction[0] < 50:
        st.markdown('<style>@keyframes blink { 50% { opacity: 0; } } .blinking { animation: blink 1s infinite; color: red; }</style>', unsafe_allow_html=True)
        st.markdown(f'<p class="blinking">Predicted Efficiency: {prediction[0]:.2f}% - Warning: Dispersant usage may not be feasible</p>', unsafe_allow_html=True)
        playsound("beep.mp3")
    else:
        st.success(f'Predicted Efficiency: {prediction[0]:.2f}%- Dispersant usage is feasible')
