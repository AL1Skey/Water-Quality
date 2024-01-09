import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns

def iscat(data):
    return True if data.nunique() < 15 else False

def app():
    st.header("Exploration Data Analysis")
    model = joblib.load('model.pkl')
    database = pd.read_csv('dataset.csv')
    feature = json.load(open('feature.json','r'))

    st.write('### See Value Counts of Target')
    plt.figure(figsize=(10,10))
    st.pyplot(database['is_safe'].value_counts().plot.barh().get_figure())
    st.write('Insight: Data are imbalance')

    st.write('### See Cardinality of Data')
    plt.figure(figsize=(10,10))
    st.pyplot(database.nunique().plot.barh(xlim=[0,20]).get_figure())
    st.write('Insight: All columns except is_safe, uranium, selenium, and mercury has large unique data with cardinality > 15')

    st.write('### See mean of viruses based on target')
    plt.figure(figsize=(10,10))
    st.pyplot(database.groupby('is_safe')['viruses'].mean().plot.barh().get_figure())
    st.write('Insight: There more viruses on not safe water')

    st.write('### See mean of aluminium based on target')
    plt.figure(figsize=(10,10))
    st.pyplot(database.groupby('is_safe')['aluminium'].mean().plot.barh().get_figure())
    st.write('Insight: Aluminium composition are higher on safe water than unsafe water.')

    #---------------------------------------------------------------------------------------------
    st.write('### See mean of ammonia based on target')
    plt.figure(figsize=(10,10))
    # raise Exception(database.groupby('is_safe')['ammonia'].mean().plot.barh())
    st.pyplot(database.groupby('is_safe')['ammonia'].mean().plot.barh().get_figure())
    st.write('Insight: Ammonia composition are higher on unsafe water than safe water.')

    st.write('### See mean of flouride based on target')
    plt.figure(figsize=(10,10))
    st.pyplot(database.groupby('is_safe')['flouride'].mean().plot.barh().get_figure())
    st.write('Insight: Flouride composition are higher on safe water than unsafe water.')

    st.write('### See Correlation of Each Data')
    fig = plt.figure(figsize=(15,20))
    sns.heatmap(database.corr(),annot=True,fmt='.2f',cmap='coolwarm')
    st.pyplot(fig)
    st.write("Insight: 'aluminium', 'arsenic', 'cadmium', 'chloramine', 'chromium', 'silver' are column that have high correlation towards data with correlation > 10%.")

    st.write('### See Skewness of Feature')
    fig2 = plt.figure(figsize=(10,10))
    sns.barplot(database[feature].select_dtypes(include=np.number).skew(),orient='h')
    st.pyplot(fig2)
    st.write('Insight: Feature has Skewwed distribution with skewness > 0.5 except cadmium.')