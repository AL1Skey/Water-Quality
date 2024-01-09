import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json

# Contents of main.py
import eda
import prediction
import home
import overview
PAGES = {
    "Home": home,
    "EDA": eda,
    "Model Showcase":prediction,
    "Model Overview":overview
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

