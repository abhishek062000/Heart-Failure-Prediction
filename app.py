from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))



app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        cpk = int(request.form['CPK'])
        plate = float(request.form['platelets'])
        sc = float(request.form['SC'])
        ss = int(request.form['SS'])
        ef = int(request.form['EF'])
        time = int(request.form['time'])
        smoke = int(request.form['Smoking'])
        anae = int(request.form['anaemia'])
        pressure = int(request.form['bloodpressure'])
        dia = int(request.form['Diabetes'])
        sex = int(request.form['Gender'])
        
        data = np.array([[age,anae,cpk,dia,ef,pressure,plate,sc,ss,sex,smoke,time]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)



if __name__ == '__main__':
	app.run(debug=True)



