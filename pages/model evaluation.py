import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json
# from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt


def right_percent(feature,target,model_train):
    result={}

    if isinstance(model_train, dict) and len(model_train)>1:
        for key,modeling in model_train.items():
            predict = modeling.predict(feature)
            result[key] = f'{round((target == predict).mean()*100,2)}%'
            st.write(f'{key} : {result[key]}')
    else:
        predict = model_train.predict(feature)
        result = f'{round((target == predict).mean()*100,2)}%'
        st.write(f'model: {result}')
    return result    

def full_eva(feature,target,model_train):
    if isinstance(model_train, dict) and len(model_train)>1:
        for key, modeling in model_train.items():
            predict = modeling.predict(feature)
            st.write("="*20,key,"="*20)
            st.write(classification_report(target,predict))
    else:
        predict = model_train.predict(feature)
        st.code(classification_report(target,predict))

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
        fig = plt.figure(figsize=(10,10))
        disp.plot()
        st.pyplot(fig)

#App    
model = joblib.load('model.pkl')
database = pd.read_csv('dataset.csv')
with open('feature.json','r') as file:
    feature = json.load(file) 

X_inf, y_inf = database.loc[:len(database)*0.2,database.columns[:-1]], database.loc[:len(database)*0.2,database.columns[-1]]
X_inf=X_inf[feature]
# Model Performance
st.write("# Model Performance")
st.write("# Classification Report")
st.code('''
Classification Report
                precision    recall  f1-score   support

           0       0.93      0.95      0.94      2835
           1       0.95      0.93      0.94      2835

    accuracy                           0.94      5670
   macro avg       0.94      0.94      0.94      5670
weighted avg       0.94      0.94      0.94      5670
             ''',language='plaintext')
st.write('Precision, Recall and F1-Score have high value')
    # full_eva(X_inf,y_inf,model)
    # Confusion Matrix
st.write("# Confusion Matrix")
st.image('boosttest.png')
st.write('This model produce 144 False Positive and 208 False Negative')
    