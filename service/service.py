#Import Flask
from flask import Flask, request, render_template
from flask_cors import CORS

from model_load import loadmodels
import numpy as np

#Initialize flask
app = Flask(__name__)

#Load models
CORS(app)
global loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2, graph
loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2, graph = loadmodels()

#Define a route
@app.route('/home/mainpage/', methods=['GET','POST'])
def main_page():
	return render_template('index.html')

@app.route('/home/', methods=['GET','POST'])
def home_page():
	return render_template('cover.html')

@app.route('/how/', methods=['GET','POST'])
def how_page():
	return render_template('how.html')

@app.route('/help/', methods=['GET','POST'])
def help_page():
	return render_template('help.html')

@app.route('/mainpage/predict/', methods=['GET','POST'])
def default():
	# print (request.data)
	# print (request.args)
	# print (request.form)
	data = None
	if request.method == 'GET':
		print ("GET Method")
		data = request.args

	if request.method == 'POST':
		print ("POST Method")
		if (request.is_json):
			data = request.get_json()

	print("Data received:", data)

	# Get data
	CreditScore = data.get("CreditScore")
	Geography = data.get("Geography")
	Gender = data.get("Gender")
	Age = data.get("Age")
	Tenure = data.get("Tenure")
	Balance = data.get("Balance")
	NumOfProducts = data.get("NumOfProducts")
	HasCrCard = data.get("HasCrCard")
	IsActiveMember = data.get("IsActiveMember")
	EstimatedSalary = data.get("EstimatedSalary")

	print ("\nCreditScore: ",CreditScore,
			"\nGeography: ", Geography,
			"\nGender: ", Gender,
			"\nAge: ", Age,
			"\nTenure: ", Tenure,
			"\nBalance: ", Balance,
			"\nNumOfProducts: ", NumOfProducts,
			"\nHasCrCard: ", HasCrCard,
			"\nIsActiveMember: ", IsActiveMember,
			"\nEstimatedSalary: ", EstimatedSalary)

	# Convert categorical data
	[Geography] = loaded_labelEncoderX1.transform([Geography])
	[Gender] = loaded_labelEncoderX2.transform([Gender])

	predict = np.array([CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary])
	print("\npredict: ", predict)
	predict = loaded_scaler.transform([predict])
	print("predict: ", predict)

	with graph.as_default():
		Exited = ""
		score = loaded_model.predict(predict)
		scr =str(round(score[0].item() * 100, 2))
		print("\nFinal score: ", score)
		leave = (score > 0.5)
		if leave:
			Exited += "Leave"
		else:
		    Exited += "Stay"
		return Exited + " | " + "Leaving Probability: " + scr + "%"

# Run
app.run(host='0.0.0.0',port=5000)

# http://localhost:5000/mainpage/predict/?CreditScore=3&Geography=France&Gender=Male&Age=36&Tenure=2&Balance=1200.34&NumOfProducts=3&HasCrCard=1&IsActiveMember=0&EstimatedSalary=120000