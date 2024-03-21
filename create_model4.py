import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

def train_model():
    # Load the dataset
    df = pd.read_csv('house_price_data.csv')

    # Convert categorical variables to binary
    df[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']] = \
        df[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']].replace(
            {'yes': 1, 'no': 0}).astype(int)

    # Convert furnishingstatus using label encoding
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

    model_accuracies = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),  
            ('model', model)
        ])
        
        # Fit the pipeline
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        model_accuracies[name] = r2

        # Save the best performing model
        joblib.dump(pipe, f'{name.lower().replace(" ", "_")}_model.pkl')

    return model_accuracies

model_accuracies = train_model()

for name, accuracy in model_accuracies.items():
    print(f'{name}: R^2 Score = {accuracy}')
