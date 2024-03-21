import subprocess
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    df = pd.read_csv('house_price_data.csv')
    df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']] = df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']].replace({'yes': 1, 'no': 0}).astype(int)
    df['furnishingstatus'] = LabelEncoder().fit_transform(df['furnishingstatus'])

    x = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    model = LinearRegression(n_jobs=5)
    pipe = Pipeline([
        ('scaler', StandardScaler()),  
        ('model', model)
    ])
    pipe.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipe, 'linear_regression_model.pkl')

    return redirect(url_for('index'))


train_model()