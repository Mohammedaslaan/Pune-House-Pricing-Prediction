import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression,Ridge
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
df = pd.read_csv('house_price_data.csv')
df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']] = df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']].replace({'yes': 1, 'no': 0}).astype(int)
df['furnishingstatus']=LabelEncoder().fit_transform(df['furnishingstatus'])

models = {
    'Linear Regression': LinearRegression(n_jobs=5),
}

x=df.drop("price",axis=1)
y=df["price"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

a=1
best_r2=-1

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),  
        ('model', model)
    ])
    
    # Fit the pipeline
    model=pipe.fit(X_train, y_train)
    y_pre=model.predict(X_test)
    
    print(f'{a} :- {name} - MSE: {mean_squared_error(y_test, y_pre):.4f}')
    print(f'     {name} - R2: {r2_score(y_test, y_pre):.4f}')
    print(f'     {name} - MAE: {mean_absolute_error(y_test, y_pre):.4f}',"\n")
    a+=1
    r2=r2_score(y_test, y_pre)        
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

# Data to predict on
data_to_predict = [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 0]]

if best_model is not None:
    prediction = best_model.predict(data_to_predict)
    print("Best Model:", best_model_name)
    print("Predicted Price:", prediction)
else:
    print("No best model found. Something went wrong.")
