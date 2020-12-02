import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pip install pipreqs
import pickle
from PIL import Image

# file location:   /Users/HB/Desktop/python/Job-internship assessments/spark_grip


st.title("Marks prediction according to hours studied")

def image_load():
    image = Image.open('study_image.jpg')
    st.image(image, caption='God help those who help themselves!',use_column_width=True)

image_load()

@st.cache(persist=True)
def load_data():
    url = "http://bit.ly/w-data"
    data = pd.read_csv(url)
    #st.dataframe(data)
    return data
data = load_data()


def plots(data):
    st.subheader("Study hours Vs Marks Scatter plot")
    fig = px.scatter(data, x = 'Hours', y = 'Scores')
    st.plotly_chart(fig)
    
plots(data)


def prediction_time(h):
    filename = 'saved_model_for_streamlit.sav'
    lr = pickle.load(open(filename, 'rb'))


    if h <= 0.00:
        st.write('You are going to fail')
    else:
        h = [[h]]
        predicted_score = lr.predict(h)
        if predicted_score >= 100:
            st.write('You are going to score 100, Enjoy life!!')
        else:
            st.write('This will be your marks:', predicted_score)
        
h = st.number_input('hours')       
prediction_time(h)        
        
        
