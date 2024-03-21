import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    # Load the dataset
    df = pd.read_csv('house_price_data.csv')

    # Convert categorical variables to binary
    df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']] = \
        df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']].replace({'yes': 1, 'no': 0}).astype(int)

    # Convert furnishingstatus using one-hot encoding
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps for numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the pipeline with preprocessing and the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression(n_jobs=5))
    ])

    # Fit the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'linear_regression_model.pkl')

    return model

trained_model = train_model()