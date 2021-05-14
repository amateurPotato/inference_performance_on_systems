#!/usr/bin/python3
from flask import Flask, render_template, request, redirect, url_for
from markupsafe import escape
import datetime
import time
from ec2_function import *

app = Flask(__name__)


@app.route('/')
def home():
    return ('OK')


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        start = time.time()
        content = request.get_json()
        #print(content)
        #print(request.get_data())
        event = ec2_handler(content)
        stop = time.time()
        event['inference_time'] = stop - start
    return (event)

if __name__ == "__main__":
    #import logging
    #logging.basicConfig(filename='/home/ak8257/error.log',level=logging.DEBUG)
    app.run(debug = True)
