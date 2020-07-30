#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:46:51 2020

@author: milan
"""

from flask import Flask, render_template, url_for, request
import pickle
import os

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__,static_folder=os.path.abspath('static'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = clf.predict(data)
    return render_template('result.html', prediction = my_prediction)

if __name__=='__main__':
    app.run(debug=True)
    
















































