from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

app = FastAPI(title="AddisProphecy LITE API")

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "housing_model_lite.json")

# Default metadata fallback in case file loading fails
DEFAULT_LOCATIONS = [
    'Bole', 'Old Airport', 'Kazanchis', 'Sarbet', 'CMC', 
    'Lebu', 'Ayat', 'Summit', 'Jemo', 'Gullele', 'Akaki', 'Piassa'
]
DEFAULT_TYPES = ['Villa', 'Apartment', 'Condominium']

# Global variables for model
model_data = {
    "locations": DEFAULT_LOCATIONS,
    "types": DEFAULT_TYPES
}

@app.on_event("startup")
def load_model():
    global model_data
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'r') as f:
                loaded_data = json.load(f)
                # Ensure we have the keys we need
                if "locations" in loaded_data and "types" in loaded_data:
                    model_data = loaded_data
                    print("Lite JSON model loaded successfully!")
                else:
                    print("Warning: JSON model missing keys, using defaults for metadata.")
                    # Still try to keep coefficients if available
                    if "coefficients" in loaded_data:
                        model_data.update(loaded_data)
        else:
            print(f"Error: Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

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
    return {"message": "AddisProphecy LITE API is online", "status": "active"}

@app.get("/metadata/housing")
def get_housing_metadata():
    return {
        "locations": model_data.get("locations", DEFAULT_LOCATIONS),
        "types": model_data.get("types", DEFAULT_TYPES)
    }

@app.post("/predict/addis-house")
def predict_house(data: AddisHouseData):
    try:
        # Default fallback if weights are missing (avoid crash)
        if "coefficients" not in model_data:
             return {
                "estimated_price_etb": "Wait - Model Loading",
                "raw_price": 0,
                "note": "Model data not fully loaded yet."
            }

        # 1. Encode Location
        loc_list = model_data.get("locations", DEFAULT_LOCATIONS)
        loc_val = loc_list.index(data.Location) if data.Location in loc_list else 0
        
        # 2. Encode Type
        type_list = model_data.get("types", DEFAULT_TYPES)
        type_val = type_list.index(data.Type) if data.Type in type_list else 0
        
        # 3. Features array: [Location, Type, Area, Bedrooms, Bathrooms, Age, Distance]
        features = [
            loc_val,
            type_val,
            data.Area,
            data.Bedrooms,
            data.Bathrooms,
            data.Age,
            data.Distance_to_Center
        ]
        
        # 4. Manual Prediction: price = intercept + sum(w * x)
        weights = model_data.get("coefficients", [])
        intercept = model_data.get("intercept", 0)
        
        prediction = intercept
        for i in range(len(features)):
            if i < len(weights):
                prediction += features[i] * weights[i]
            
        return {
            "estimated_price_etb": f"{max(0, prediction):,.2f}",
            "raw_price": max(0, prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Local dev
    uvicorn.run(app, host="0.0.0.0", port=8000)
