import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json

st.header("Water Quality Prediction")
st.image('default.png')
# Data Loading
model = joblib.load('model.pkl')
df = pd.read_csv('dataset.csv')
feature = json.load(open('feature.json','r'))
# Header
st.write("# Data Frame")
st.write(df)

# Model Showcase
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

# Conclusion
st.write("# Conclusion")
st.write("""Gradient Boosting are best model because it has high precision and f1-score with lowest False Positive error.

This model has weakness that it'll produce False Positive when it meet:
- aluminium data in range of 1.205 - 3.705
- arsenic data in range of 0.02 - 0.04
- cadmium data in range of 0.002 - 0.007
- chloramine data in range of 2.535 - 6.08
- chromium data in range of 0.2 - 0.645
- silver data in range of 0.08 - 0.36""")