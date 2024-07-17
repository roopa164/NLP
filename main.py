import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense


# load the pretrained model
from keras import models
model = models.load_model('simple_rnn_imdb.h5')

# preprocess the input text
# load the IMDB dataset word Index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}



# for each word find decoder 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)  for word in words]
    padded_review = sequence.pad_sequences([encoded_review] , maxlen= 500)
    return padded_review


# streamlit
import streamlit as st

st.title("IMDB MOVIE REVIEW Sentiment Analysis")
st.write('Enter a Movie review to classify it as positive or negative')

#User_input
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >0.3 else 'Negative'

    # Display the result
    print("Sen----",sentiment)
    st.write(f'Sentiment:{ sentiment }')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review')



