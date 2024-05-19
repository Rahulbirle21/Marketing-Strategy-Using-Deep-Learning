import pandas as pd
import numpy as np
import streamlit as st
from tensorflow import keras
from keras.models import load_model


html_temp = """ 
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Welcome to Outlet Sales Prediction </h2>
</div> <br/>"""
st.markdown(html_temp,unsafe_allow_html=True)

model = load_model('ANN_model.h5')
df = pd.read_csv('After_EDA.csv')

affl = st.slider('Select Demographic Affluence',0,16)
age = st.slider('Select Age of the customer',0,80)
cluster = st.selectbox('Select Cluster',df['DemClusterGroup'].unique())
gender = st.selectbox('Select Gender of the customer',df['DemGender'].unique())
class_ = st.selectbox('Select Loyalty class',df['LoyalClass'].unique())
spending = st.slider('Select Spendings',0,15000)
time = st.slider('Select Loyalty Time',0,15)
region = st.selectbox('Select Region',df['Region'].unique())

input = pd.DataFrame([[affl,age,cluster,gender,class_,spending,time,region]],columns=['Affluent','Age','Cluster','Gender','Class','Spending','Time','Region'])
input['Cluster'] = input['Cluster'].replace(['A','B','C','D','E','F','U'],[1,4,6,5,2,3,7])
input['Gender'] = input['Gender'].replace(['M','F','U'],[2,3,1])
input['Region'] = input['Region'].replace(['London','Midlands','Bristol','Cardiff','Birmingham','Wales','Yorkshire',
                                      'Bolton','Nottingham','Leicester','Scotland','Ulster','Trafford'],[13,12,11,10,9,8,7,6,5,4,3,2,1])
input['Class'] = input['Class'].replace(['Silver','Tin','Gold','Platinum'],[4,3,2,1])

if st.button('Identify Customer'):

    pred = np.round(model.predict([input]),2)
    if pred >= 0.76:

        st.write('The user will buy the product and generate profit for the company')
    else:
        st.write('The user will not buy the product and the company will face loss')