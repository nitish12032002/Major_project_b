from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re
from tensorflow import keras
import streamlit as st
from textblob import TextBlob
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = keras.models.load_model('sentiment analysis.h5')
df = pd.read_csv('filename.csv')
Data_tweet = df[['tweet']]


def text_cleaning(text):
  text = re.sub(r'@[A-Za-z0-9]+', '', text)     # removing @mentions
  text = re.sub(r'@[A-Za-zA-Z0-9]+', '', text)  # removing @mentions
  text = re.sub(r'@[A-Za-z]+', '', text)        # removing @mentions
  text = re.sub(r'@[-)]+', '', text)            # removing @mentions
  text = re.sub(r'#', '', text)                # removing '#' sign
  text = re.sub(r'RT[\s]+', '', text)           # removing RT
  text = re.sub(r'https?\/\/\S+', '', text)     # removing the hyper link
  text = re.sub(r'&[a-z;]+', '', text)          # removing '&gt;'

  return text


Data_tweet['tweet'] = Data_tweet['tweet'].apply(text_cleaning)


def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
  return TextBlob(text).sentiment.polarity


Data_tweet['Subjectivity'] = Data_tweet['tweet'].apply(getSubjectivity)
Data_tweet['Polarity'] = Data_tweet['tweet'].apply(getPolarity)
print(Data_tweet)


def getAnalysis(score):
  if score <= 0:
    return 'Negative'
  else:
    return 'Positive'


Data_tweet['Analysis'] = Data_tweet['Polarity'].apply(getAnalysis)


x_train, x_test, y_train, y_test = train_test_split(
    Data_tweet['tweet'].values, Data_tweet['Analysis'].values, test_size=0.30)

# converting the strings into integers using Tokenizer
# instantiating the tokenizer
max_vocab = 20000000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(x_train)

# checking the word index and find out the vocabulary of the dataset
wordidx = tokenizer.word_index
V = len(wordidx)
print('The size of datatset vocab is: ', V)

# converting tran and test sentences into sequences
train_seq = tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)
print('Training sequence: ', train_seq[0])
print('Testing sequence: ', test_seq[0])

# padding the sequences to get equal length sequence because its conventional to use same size sequences
# padding the traing sequence
pad_train = pad_sequences(train_seq)
T = pad_train.shape[1]
print('The length of training sequence is: ', T)

# padding the test sequence
pad_test = pad_sequences(test_seq, maxlen=T)
print('The length of testing sequence is: ', pad_test.shape[1])


def predict_sentiment(text):
  # preprocessing the given text
  text_seq = tokenizer.texts_to_sequences(text)
  text_pad = pad_sequences(text_seq, maxlen=T)

  # predicting the class
  predicted_sentiment = model.predict(text_pad).round()

  if predicted_sentiment == 1.0:
    result = 'It is a positive sentiment'
    print('It is a negative sentiment')
  else:
    result = 'It is a negative sentiment'
    print('It is a negative sentiment')

  return result


html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center>
   <center><p style="font-size:25px;color:white;margin-top:10px;"Major Project 2022</p></center>
   </div>
   </div>
   </div>
   """
st.markdown(html_temp, unsafe_allow_html=True)
st.header("Russia v/s Ukraine Twitter Sentiment analysis LSTM System ")


text = st.text_area("Write Twitter Review")

if st.button("Twitter Sentiment Analysis"):
  result=predict_sentiment([text])
  st.success('Model has predicted {}'.format(result))

#if st.button("About"):
#  st.subheader("Developed by Nitish Nama")
#  st.subheader("Student , Department of Computer Engineering")
html_temp="""
   <div class="" style="background-color:orange;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Major 2022 Project Deployment</p></center>
   </div>
   </div>
   </div>
   """
st.markdown(html_temp, unsafe_allow_html=True)
