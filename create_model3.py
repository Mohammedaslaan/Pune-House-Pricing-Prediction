import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import r2_score

def train_model():
    # Load the dataset
    df = pd.read_csv('house_price_data.csv')

    # Convert categorical variables to binary
    df[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']] = \
        df[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']].replace(
            {'yes': 1, 'no': 0}).astype(int)

    # Convert furnishingstatus using one-hot encoding
    # df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    df['furnishingstatus'] = LabelEncoder().fit_transform(df['furnishingstatus'])

    models = {
        'Linear Regression': LinearRegression(n_jobs=5),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Ridge Regression': Ridge()
    }
    
    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_r2 = -1
    best_model = None

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),  
            ('model', model)
        ])
        
        # Fit the pipeline
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = pipe

    # Save the best performing model
    joblib.dump(best_model, 'linear_regression_model.pkl')

    return best_model

trained_model = train_model()
