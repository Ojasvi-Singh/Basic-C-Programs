import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load and clean dataset
df = pd.read_csv(r"C:\Users\singh\college assistent\medical\diabetic_data.csv")

# Drop unnecessary columns
df = df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1)
df = df.replace('?', pd.NA).dropna()

# Keep only the features you're using in the UI + target
selected_features = [
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'number_diagnoses', 'insulin'
]

df = df[selected_features + ['readmitted']]

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split data
X = df[selected_features]
y = df['readmitted']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and features
with open('model.pkl', 'wb') as f:
    pickle.dump((model, X.columns.tolist()), f)

print("✅ Model trained and saved with simplified feature set.")
