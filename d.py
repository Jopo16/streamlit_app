#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection, preprocessing


# In[2]:


# loading in the model to predict on the data
with open('DTC.model', 'rb') as fp:
        classifier = pickle.load(fp)


# In[3]:
columns_list=("WindSpeed9am", "WindSpeed3pm", "WindGustSpeed", "Temp3pm", "Temp9am", "MinTemp", "MaxTemp", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm")
def checkNumeric(l):
    return all(type(e) in (int, float) for e in l)
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

# In[4]:


def main():
      # giving the webpage a title
    st.title("Rain Forecast Prediction Classifier")
    st.markdown('<style>h1{color: white;}</style>', unsafe_allow_html=True)
    #st.header("Data mining course 2021")
    #st.markdown('<style>h2{color: white;}</style>', unsafe_allow_html=True)

    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <style>
    .stApp {
    background-image: url("https://www.wallpaperuse.com/wallp/44-443614_m.jpg");
    background-size: cover;
    }
    </style>
    <div style ="background-color:dimgrey;padding:13px"> 
    <h2 style ="color:white;text-align:center;">Streamlit DTC App</h2>
    </div>
    """
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    # the data required to make the prediction
    Windspeed9am = st.text_input("WindSpeed9am", "Type Here") 
    Windspeed3pm = st.text_input("WindSpeed3pm", "Type Here")
    WindgustSpeed = st.text_input("WindGustSpeed", "Type Here")
    Temp3atpm = st.text_input("Temp3pm", "Type Here")
    Temp9atam = st.text_input("Temp9am", "Type Here")
    MinTempt = st.text_input("MinTemp", "Type Here")
    MaxTempt = st.text_input("MaxTemp", "Type Here")
    Humidity9amt = st.text_input("Humidity9am", "Type Here")
    Humidity3pmt = st.text_input("Humidity3pm", "Type Here")
    Pressure9amt = st.text_input("Pressure9am", "Type Here")
    Pressure3pmt = st.text_input("Pressure3pm", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        y=(Windspeed9am, Windspeed3pm, WindgustSpeed, Temp3atpm, Temp9atam, MinTempt,MaxTempt, Humidity9amt, Humidity3pmt, Pressure9amt, Pressure3pmt)
        if checkNumeric(y)==False:
            st.error("Please enter all numeric values")
        else:    
            df=pd.DataFrame([[Windspeed9am, Windspeed3pm, WindgustSpeed, Temp3atpm, Temp9atam, MinTempt,MaxTempt, Humidity9amt, Humidity3pmt, Pressure9amt, Pressure3pmt]], columns=columns_list)
            df=df.fillna(0)
            X = scaler.fit_transform(df)
            result = classifier.predict(X)
            if result.flat[0]==0:
                st.success("Tomorrow it will not rain")
            else:
                st.success("Tomorrow it will rain")
     
if __name__=='__main__':
    main()





