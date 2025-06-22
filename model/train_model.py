import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('../../data/data_core_updated.csv')
df.columns = df.columns.str.strip()  # clean column names

# Encode categorical variables
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_fert = LabelEncoder()

df['Crop'] = le_crop.fit_transform(df['Crop'])
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Fertilizer'] = le_fert.fit_transform(df['Fertilizer'])

# Define features and target
X = df[['Temperature', 'Humidity', 'Soil Moisture', 'Soil Type', 'Crop',
        'Nitrogen', 'Phosphorus', 'Potassium']]
y = df['Fertilizer']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'fertilizer_model.pkl')
joblib.dump(le_crop, 'crop_encoder.pkl')
joblib.dump(le_soil, 'soil_encoder.pkl')
joblib.dump(le_fert, 'fertilizer_encoder.pkl')

print("✅ Model training complete.")
