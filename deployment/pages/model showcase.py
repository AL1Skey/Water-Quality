import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json

# Data Loading
model = joblib.load('model.pkl')
df = pd.read_csv('dataset.csv')
feature = json.load(open('feature.json','r'))

#func

st.write('# Model Showcase')
st.write('')
def user_input():
    inputter = {}
    col = st.columns([3,3])
    
    for val,i in enumerate(feature): 
        minimum = df[i].min() - (df[i].min()*0.2)
        maximum = df[i].max() + (df[i].max()*0.2)
        inputter[i] = col[val%2].slider(i,minimum,maximum)
    inputter = pd.DataFrame([inputter])
    return inputter
# Sidebar Input
input = user_input()
st.subheader('Input')
st.write(input)
if st.button('predict'):
    prediction = model.predict(input)
    
    if prediction == 0:
        prediction = 'Not Safe'
    else:
        prediction = 'Safe'
    
    st.header(f'This water are {prediction} to drink')