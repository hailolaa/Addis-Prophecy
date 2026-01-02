import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = 'd:/Project/Ml/ethio_ml_hub/models/'
DATA_DIR = 'd:/Project/Ml/ethio_ml_hub/data/'

def train_addis_housing_refined():
    print("Training Refined Addis Housing model...")
    csv_path = os.path.join(DATA_DIR, 'addis_housing.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Encoders for categorical features
    le_loc = LabelEncoder()
    df['Location'] = le_loc.fit_transform(df['Location'])
    
    le_type = LabelEncoder()
    df['Type'] = le_type.fit_transform(df['Type'])
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    joblib.dump(model, os.path.join(MODEL_DIR, 'addis_house_model.pkl'))
    joblib.dump(le_loc, os.path.join(MODEL_DIR, 'addis_location_encoder.pkl'))
    joblib.dump(le_type, os.path.join(MODEL_DIR, 'addis_type_encoder.pkl'))
    
    # Also save metadata for the frontend
    metadata = {
        'locations': list(le_loc.classes_),
        'types': list(le_type.classes_)
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, 'housing_metadata.pkl'))
    
    print("Refined Addis Housing model and encoders saved.")

if __name__ == "__main__":
    train_addis_housing_refined()
