import pickle
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('index.html')
    # return"<h1>Hello World! <br> &nbsp; &nbsp; &nbsp; &nbsp; --ShivamDevHere</h1>"

if __name__=="__main__":
    app.run(host="0.0.0.0")