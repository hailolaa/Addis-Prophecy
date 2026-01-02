from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn

app = FastAPI(title="EthioML Hub Unified API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Global variables for models
models = {}

@app.on_event("startup")
def load_models():
    try:
        # Load Housing models
        models["addis_house"] = joblib.load(os.path.join(MODELS_DIR, 'addis_house_model.pkl'))
        models["addis_loc_enc"] = joblib.load(os.path.join(MODELS_DIR, 'addis_location_encoder.pkl'))
        models["addis_type_enc"] = joblib.load(os.path.join(MODELS_DIR, 'addis_type_encoder.pkl'))
        models["housing_meta"] = joblib.load(os.path.join(MODELS_DIR, 'housing_metadata.pkl'))
        
        # Load others (optional, keeping for backward compatibility)
        if os.path.exists(os.path.join(MODELS_DIR, 'titanic_lr_model.pkl')):
            models["titanic_lr"] = joblib.load(os.path.join(MODELS_DIR, 'titanic_lr_model.pkl'))
            models["titanic_dt"] = joblib.load(os.path.join(MODELS_DIR, 'titanic_dt_model.pkl'))
        if os.path.exists(os.path.join(MODELS_DIR, 'iris_rf_model.pkl')):
            models["iris_rf"] = joblib.load(os.path.join(MODELS_DIR, 'iris_rf_model.pkl'))
            
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

class AddisHouseData(BaseModel):
    Location: str
    Type: str
    Area: float
    Bedrooms: int
    Bathrooms: int
    Age: int
    Distance_to_Center: float

@app.get("/")
def read_root():
    return {"message": "EthioML Hub Unified API is online"}

@app.get("/metadata/housing")
def get_housing_metadata():
    return models.get("housing_meta", {"locations": [], "types": []})

@app.post("/predict/addis-house")
def predict_house(data: AddisHouseData):
    try:
        # Encode Location
        loc_classes = list(models["addis_loc_enc"].classes_)
        loc_val = loc_classes.index(data.Location) if data.Location in loc_classes else 0
        
        # Encode Type
        type_classes = list(models["addis_type_enc"].classes_)
        type_val = type_classes.index(data.Type) if data.Type in type_classes else 0
            
        feat = pd.DataFrame([[loc_val, type_val, data.Area, data.Bedrooms, data.Bathrooms, data.Age, data.Distance_to_Center]],
                           columns=['Location', 'Type', 'Area', 'Bedrooms', 'Bathrooms', 'Age', 'Distance_to_Center'])
        
        price = float(models["addis_house"].predict(feat)[0])
        return {
            "estimated_price_etb": f"{price:,.2f}",
            "raw_price": price
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keeping other endpoints for internal testing if needed
class TitanicData(BaseModel):
    Pclass: int; Sex: str; Age: float; SibSp: int; Parch: int; Fare: float; Embarked: str
@app.post("/predict/titanic")
def predict_titanic(data: TitanicData):
    # Simplified placeholder or implementation
    return {"message": "Endpoint preserved for internal use"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
