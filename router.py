# Import the Libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib
import os
# from jinja2.utils import escape
# the function I craeted to process the data in utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
# from Untitled6 import preprocess_new
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Intialize the Flask APP
app = Flask(__name__)

# Loading the Model
# model = joblib.load('model.pkl')

with open('model1.pkl', 'rb') as file:
    model = pickle.load(file)




@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction
        review = request.form['review']

        X_new =[review]
        print (X_new)



        with open('X_tra_tfidf.pkl', 'rb') as file:
            tfidf_vectorizer = pickle.load(file)

        # Perform transformation on the test data using the loaded vectorizer
        x = tfidf_vectorizer.transform(X_new)
        print(x)
        # Make predictions on the transformed test data using the loaded model
        y_pred_test = model.predict(x)


        y_pred_new = y_pred_test

        return render_template('predict.html', labell=y_pred_test)
    else:
        return render_template('predict.html')


# Route for About page
@app.route('/about')
def about():
    return render_template('about.html')


# Run the App from the Terminal
if __name__ == '__main__':
    app.run(debug=True)