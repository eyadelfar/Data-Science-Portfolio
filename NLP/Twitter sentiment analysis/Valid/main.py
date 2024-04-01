import os
from os.path import dirname, join, realpath
import pickle as pk
import uvicorn
from fastapi import FastAPI
import numpy as np 


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

# load the sentiment model
with open("vectortizer.sav", "rb") as f:
    vect = pk.load(f)
with open("LR_Model88.sav", "rb") as f:
     model = pk.load(f)

@app.get("/predict-review")
def predict_sentiment(review: str):
    review = review.lower()
    review = word_tokenize(review.translate(str.maketrans('', '', string.punctuation)))
    stop_words = set(stopwords.words('english'))
    review = [word for word in review if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    X = review
    X = vect.transform([X])
    X = X.toarray()
    result = model.predict(X)
    output = int(result)
    probas = model.predict_proba(X)
    output_probability = "{:.2f}".format(float(probas[:, output]))
   
   # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
   
   # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result

