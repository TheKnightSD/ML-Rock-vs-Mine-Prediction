import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("sonar.csv", header=None)

X = data.drop(columns=60)
y = data[60]

# -------------------------------
# Label Encoding
# -------------------------------
label = LabelEncoder()
y = label.fit_transform(y)

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Save Pickle Files (IMPORTANT)
# -------------------------------
pickle.dump(model, open("sonar_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label, open("label_encoder.pkl", "wb"))

print("All pickle files saved successfully")
