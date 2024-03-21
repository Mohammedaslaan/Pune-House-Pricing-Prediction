
import subprocess
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_graphs', methods=['POST'])
def generate_graphs():
    # area = request.form['area']
    # bedrooms = request.form['bedrooms']
    # bathrooms = request.form['bathrooms']
    # stories = request.form['stories']
    # mainroad = request.form['mainroad']
    # guestroom = request.form['guestroom']
    # basement = request.form['basement']
    # hotwaterheating = request.form['hotwaterheating']
    # airconditioning = request.form['airconditioning']
    # parking = request.form['parking']
    # prefarea = request.form['prefarea']
    # furnishingstatus = request.form['furnishingstatus']

    # Execute the Python script to generate graphs
    # subprocess.run(['python', 'generate_graphs.py', area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus])
    subprocess.run(['python', 'generate_graphs.py'])
    return redirect(url_for('show_graphs'))

@app.route('/graphs')
def show_graphs():
    return render_template('graphs.html')



# @app.route('/predict_price', methods=['POST'])
# def predict_price():
#     # Load the trained model
#     model = joblib.load('linear_regression_model.pkl')
#     # Get input values from the form
#     area = float(request.form['area'])
#     bedrooms = int(request.form['bedrooms'])
#     bathrooms = int(request.form['bathrooms'])
#     stories = int(request.form['stories'])
#     mainroad = int(request.form['mainroad'])
#     guestroom = int(request.form['guestroom'])
#     basement = int(request.form['basement'])
#     hotwaterheating = int(request.form['hotwaterheating'])
#     airconditioning = int(request.form['airconditioning'])
#     parking = int(request.form['parking'])
#     prefarea = int(request.form['prefarea'])
#     furnishingstatus = int(request.form['furnishingstatus'])
#     # Add more input fields as needed
    
#     # Perform prediction using the model
#     data_to_predict = [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]]
#     predicted_price = model.predict(data_to_predict)[0]

#     # Pass the predicted price to the results page
#     return redirect(url_for('show_results', predicted_price=predicted_price))

# @app.route('/predict_price', methods=['POST'])
# def predict_price():
#     # Load the trained model
#     model = joblib.load('linear_regression_model.pkl')

#     # Get input values from the form
#     area = float(request.form['area'])
#     bedrooms = int(request.form['bedrooms'])
#     bathrooms = int(request.form['bathrooms'])
#     stories = int(request.form['stories'])
#     mainroad = int(request.form['mainroad'])
#     guestroom = int(request.form['guestroom'])
#     basement = int(request.form['basement'])
#     hotwaterheating = int(request.form['hotwaterheating'])
#     airconditioning = int(request.form['airconditioning'])
#     parking = int(request.form['parking'])
#     prefarea = int(request.form['prefarea'])
#     furnishingstatus = int(request.form['furnishingstatus'])
    
#     # Prepare the input data for prediction
#     data_to_predict = [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]]

#     # Perform prediction using the model
#     predicted_price = model.predict(data_to_predict)[0]

#     # Pass the predicted price to the results page
#     return redirect(url_for('show_results', predicted_price=predicted_price))
@app.route('/predict_price', methods=['POST'])
def predict_price():
    # Load the trained model
    model = joblib.load('linear_regression_model.pkl')

    # Get input values from the form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = int(request.form['mainroad'])
    guestroom = int(request.form['guestroom'])
    basement = int(request.form['basement'])
    hotwaterheating = int(request.form['hotwaterheating'])
    airconditioning = int(request.form['airconditioning'])
    parking = int(request.form['parking'])
    prefarea = int(request.form['prefarea'])
    furnishingstatus = int(request.form['furnishingstatus'])
    
    # Prepare the input data for prediction
    data_to_predict = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    # Perform prediction using the model
    predicted_price = model.predict(data_to_predict)[0]

    # Pass the predicted price to the results page
    return redirect(url_for('show_results', predicted_price=predicted_price))



@app.route('/results/<predicted_price>')
def show_results(predicted_price):
    return render_template('results.html', predicted_price=predicted_price)



if __name__ == '__main__':
    app.run(debug=True)
