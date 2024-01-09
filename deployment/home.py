import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json

def right_percent(feature,target,model_train):
    result={}

    if isinstance(model_train, dict) and len(model_train)>1:
        for key,modeling in model_train.items():
            predict = modeling.predict(feature)
            result[key] = f'{round((target == predict).mean()*100,2)}%'
            print(f'{key} : {result[key]}')
    else:
        predict = model_train.predict(feature)
        result = f'{round((target == predict).mean()*100,2)}%'
        print(f'model: {result}')
    return result    

def full_eva(feature,target,model_train):
    if isinstance(model_train, dict) and len(model_train)>1:
        for key, modeling in model_train.items():
            predict = modeling.predict(feature)
            print("="*20,key,"="*20)
            print(classification_report(target,predict))
    else:
        predict = model_train.predict(feature)
        print(classification_report(target,predict))

def conf_matrix_show(feature,target,model_train):
    if isinstance(model_train, dict) and len(model_train)>1:
        for key, modeling in model_train.items():
            prediction = modeling.predict(feature)
            cm = confusion_matrix(target,prediction)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.title(key)
            plt.show()
    else:
        prediction = model_train.predict(feature)
        cm = confusion_matrix(target,prediction)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()

def app():
    st.header("Water Quality Prediction")
    st.image('default.png')
    # Data Loading
    model = joblib.load('model.pkl')
    df = pd.read_csv('dataset.csv')
    feature = json.load(open('feature.json','r'))

    # Header
    st.write("# Data Frame")
    st.write(df)
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