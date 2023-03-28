#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import streamlit as st
import pandas as pd
import numpy as np
#import logging
#logger = tf.get_logger()
#logger.setLevel(logging.ERROR)


celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)



model2 = Sequential(Dense(units=4, activation='relu', input_shape=[1]))
model2.add(Dense(units=4, activation='relu'))
model2.add(Dense(units=1, activation='linear'))
model2.compile(loss='mean_squared_error', optimizer=Adam(0.1))
model2.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

social_acc = ['About', 'Kaggle', 'Medium', 'LinkedIn']
social_acc_nav = st.sidebar.selectbox('About', social_acc)
if social_acc_nav == 'About':
    st.sidebar.markdown("<h2 style='text-align: center;'> Salami Lukman Bayonle</h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''
    • Data Analytics/Scientist (Python/R/SQL/Tableau) \n 
    • Maintenance Specialist (Nigerian National Petroleum Company Limited) \n 
    • IBM/GOOGLE/DATACAMP Certified Data Analyst and Data Scientist''')
    st.sidebar.markdown("[ Visit Github](https://github.com/bayonlelukmansalami)")

elif social_acc_nav == 'Kaggle':
    st.sidebar.image('kaggle.jpg')
    st.sidebar.markdown("[Kaggle](https://www.kaggle.com/bayonlesalami)")

elif social_acc_nav == 'Medium':
    st.sidebar.image('medium.jpg')
    st.sidebar.markdown("[Click to read my blogs](https://medium.com/@bayonlelukmansalami/)")

elif social_acc_nav == 'LinkedIn':
    st.sidebar.image('linkedin.jpg')
    st.sidebar.markdown("[Visit LinkedIn account](https://www.linkedin.com/in/salamibayonlelukman/)")
    



st.title('Converting Celsius to Fahrenheit Web App')

Celsius = st.number_input('Temperature in Celsius')

#Celsius  = np.array(Celsius)

st.table(Celsius)


if st.button('Predict'):
    prediction = np.round(model2.predict([Celsius]), 0)
    
    st.write('Predicted Credit Score = ', prediction[0][0])



