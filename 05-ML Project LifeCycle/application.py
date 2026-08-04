import pickle
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/redgecv.pkl', 'rb'))
std_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')
    # return"<h1>Hello World! <br> &nbsp; &nbsp; &nbsp; &nbsp; --ShivamDevHere</h1>"

@app.route("/fire")
def fire():
    return render_template('fire.html')


@app.route("/fire",methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled = std_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        result = ridge_model.predict(new_data_scaled)

        return render_template('fire.html', results=result[0])

    else:
        return render_template('fire.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")