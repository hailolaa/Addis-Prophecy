import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Use relative paths for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'api', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'api', 'data')

def train_addis_housing_lite():
    print("Training Lite Addis Housing model (JSON output)...")
    
    # Check both potential data locations
    csv_path = os.path.join(DATA_DIR, 'addis_housing.csv')
    if not os.path.exists(csv_path):
        # Fallback to root data dir if api/data doesn't exist yet
        csv_path = os.path.join(BASE_DIR, 'data', 'addis_housing.csv')
        
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run generate_addis_data.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Encoders
    le_loc = LabelEncoder()
    df['Location_Enc'] = le_loc.fit_transform(df['Location'])
    
    le_type = LabelEncoder()
    df['Type_Enc'] = le_type.fit_transform(df['Type'])
    
    # Features: Location, Type, Area, Bedrooms, Bathrooms, Age, Distance
    X = df[['Location_Enc', 'Type_Enc', 'Area', 'Bedrooms', 'Bathrooms', 'Age', 'Distance_to_Center']]
    y = df['Price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    # Export weights and metadata to JSON for a dependency-free backend
    model_data = {
        'intercept': float(model.intercept_),
        'coefficients': model.coef_.tolist(),
        'feature_names': X.columns.tolist(),
        'locations': le_loc.classes_.tolist(),
        'types': le_type.classes_.tolist()
    }
    
    json_path = os.path.join(MODEL_DIR, 'housing_model_lite.json')
    with open(json_path, 'w') as f:
        json.dump(model_data, f, indent=4)
    
    print(f"LITE model saved to {json_path}")
    print("This JSON model does not require sklearn for prediction!")

if __name__ == "__main__":
    train_addis_housing_lite()
