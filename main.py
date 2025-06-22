from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ✅ Allow CORS from frontend (http://127.0.0.1:5500 if using Live Server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace * with specific origin: ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model = joblib.load("model/fertilizer_model.pkl")
crop_encoder = joblib.load("model/crop_encoder.pkl")
soil_encoder = joblib.load("model/soil_encoder.pkl")
fertilizer_encoder = joblib.load("model/fertilizer_encoder.pkl")

class InputData(BaseModel):
    Temperature: float
    Humidity: float
    SoilMoisture: float
    SoilType: str
    Crop: str
    Nitrogen: int
    Phosphorus: int
    Potassium: int

@app.post("/predict")
def predict(data: InputData):
    crop_encoded = crop_encoder.transform([data.Crop])[0]
    soil_encoded = soil_encoder.transform([data.SoilType])[0]

    features = np.array([[
        data.Temperature,
        data.Humidity,
        data.SoilMoisture,
        soil_encoded,
        crop_encoded,
        data.Nitrogen,
        data.Phosphorus,
        data.Potassium
    ]])

    prediction = model.predict(features)[0]
    fertilizer = fertilizer_encoder.inverse_transform([prediction])[0]

    organic_alt = ""
    if data.Nitrogen < 50:
        organic_alt = "Use Vermicompost or Cow Dung"
    elif data.Phosphorus < 30:
        organic_alt = "Use Bone Meal or Rock Phosphate"
    elif data.Potassium < 40:
        organic_alt = "Use Banana Peel Compost"

    return {
        "Inorganic Recommendation": fertilizer,
        "Organic Alternative": organic_alt
    }
