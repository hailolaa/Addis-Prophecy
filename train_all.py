import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

MODEL_DIR = 'd:/Project/Ml/ethio_ml_hub/models/'
DATA_DIR = 'd:/Project/Ml/ethio_ml_hub/data/'

def train_titanic():
    print("Training Titanic models...")
    # Using small dummy set for simplicity if raw file not available, 
    # but let's try to find it or create standard one
    data = {
        'Pclass': [1, 2, 3, 1, 3],
        'Sex': ['male', 'female', 'female', 'male', 'female'],
        'Age': [22, 38, 26, 35, 35],
        'SibSp': [1, 1, 0, 0, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05],
        'Embarked': ['S', 'C', 'S', 'S', 'S'],
        'Survived': [0, 1, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    lr = LogisticRegression()
    lr.fit(X, y)
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    
    joblib.dump(lr, os.path.join(MODEL_DIR, 'titanic_lr_model.pkl'))
    joblib.dump(dt, os.path.join(MODEL_DIR, 'titanic_dt_model.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'titanic_sex_encoder.pkl'))
    print("Titanic models saved.")

def train_iris():
    print("Training Iris model...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    rf = RandomForestClassifier()
    rf.fit(X, y)
    
    joblib.dump(rf, os.path.join(MODEL_DIR, 'iris_rf_model.pkl'))
    print("Iris model saved.")

def train_addis_housing():
    print("Training Addis Housing model...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'addis_housing.csv'))
    
    # Simple encoding for location
    le = LabelEncoder()
    df['Location'] = le.fit_transform(df['Location'])
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(MODEL_DIR, 'addis_house_model.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'addis_location_encoder.pkl'))
    print("Addis Housing model saved.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    train_titanic()
    train_iris()
    train_addis_housing()
    print("All models trained and exported successfully!")
