#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib
import os
from xgboost import XGBRegressor



# In[26]:


XG = joblib.load(r'C:\Users\anilr\XG.pkl')

sd = StandardScaler()
#sd = joblib.load(r'C:\Users\anilr\XG.pkl')
# In[29]:



training_data = pd.read_csv(r'D:\ExcelR Data Science\Project\temperature_data.csv')
sd.fit(training_data.drop(columns=['motor_speed']))

import base64



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
        font-family: 'Arial', sans-serif;
        font-style: italic;
    
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
def main():
   
    add_bg_from_local('bg2.jpg') 
    logo_path = "https://learn.excelr.com/static/images/logo.3e4e292b8110.png"  
    st.sidebar.image(logo_path, use_column_width=True)

  # Increase font size and change text color in the title and description
    st.markdown(
        f"""
        <h2 style="font-size: 40px; color: #FF0000;">Motor Speed Prediction Model</h1>
        <p style="font-size: 18px; color: #FFFF00;">This model uses an XGBoost model to predict motor speed based on the given input features.</p>
        """,
        unsafe_allow_html=True,
    )    
#st.title("Motor Speed Prediction App")
  #  st.write("This app uses an XGBoost model to predict motor speed based on the given input features.")


    # User Input Section
    st.sidebar.title("Input Features")
    
    # Create input widgets for each feature (you can add more if needed)
    ambient = st.sidebar.slider("Ambient Temperature", min_value=0.0, max_value=100.0, step=0.1)
    coolant = st.sidebar.slider("Coolant Temperature", min_value=0.0, max_value=100.0, step=0.1)
    u_d = st.sidebar.slider("D_Current Component ", min_value=0.0, max_value=100.0, step=0.1)
    u_q = st.sidebar.slider("Q_Current Component", min_value=0.0, max_value=100.0, step=0.1)
    torque = st.sidebar.slider("Torque", min_value=0.0, max_value=1000.0, step=1.0)
    i_d = st.sidebar.slider("D_Voltage Component ", min_value=0.0, max_value=100.0, step=0.1)
    i_q = st.sidebar.slider("Q_Voltage Component", min_value=0.0, max_value=100.0, step=0.1)
    pm =  st.sidebar.slider("Peramant Magnitude", min_value=0.0, max_value=100.0, step=0.1)
    stator_tooth =  st.sidebar.slider("Statr Tooth", min_value=0.0, max_value=100.0, step=0.1)
    stator_yoke =  st.sidebar.slider("Stator_Yoke", min_value=0.0, max_value=100.0, step=0.1) 
    

    # Prepare the user input data for prediction
    user_input = pd.DataFrame({
        'ambient': [ambient],
        'coolent': [coolant],
        'u_d':[u_d],
        'u_q':[u_q],
        'torque':[torque],
        'i_d':[i_d],
        'i_q':[i_q],
        'pm':[pm],
        'stator_tooth':[stator_tooth],
        'stator_yoke':[stator_yoke]
    })

    # Standardize the user input data using the same scaler used for the training data
    user_input_scaled = sd.fit_transform(user_input)
    
   # Print user input and scaled input for debugging
    st.write("User Input:")
    st.write(user_input)
   


    if st.button('Predict'):
        prediction = XG.predict(user_input)
        #st.write('Prediction:', prediction)
      # Display the prediction
        st.write("Predicted Motor Speed:", prediction[0])

      # Add the file upload button
    uploaded_file = st.file_uploader("Upload a file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded file (assuming it's a CSV) and display its contents
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded File:")
        st.write(df)

        df_scaled = sd.transform(df)
        df['Predicted_Speed'] = XG.predict(df_scaled)

        # Save the results to a new CSV file
        results_file = os.path.join('D:\ExcelR Data Science\Project', 'predicted_results.csv')
        df.to_csv(results_file, index=False)
        st.success(f"Predicted results saved to: {results_file}")

if __name__ == "__main__":
    sd = StandardScaler()
    main()






