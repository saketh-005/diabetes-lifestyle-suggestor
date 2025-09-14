import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('dataset/synthetic_amd_federated_dataset.csv')

# Select features for diabetes risk prediction (example selection)
features = [
    'Age', 'Sex', 'Ethnicity', 'Smoking_Status', 'Alcohol_Level', 'BMI', 'Physical_Activity',
    'Family_History', 'Hypertension', 'Cardio_Disease', 'Medication', 'HDL', 'LDL', 'Cholesterol',
    'Triglycerides', 'HbA1c', 'CMT', 'RPE_Elevation', 'IRF', 'SRF', 'ILM_RPE', 'Drusen_Volume',
    'Drusen_Reflectivity', 'EZ_Integrity', 'Choroidal_Thickness', 'HRF_Count', 'Texture',
    'Disc_Cup_Ratio', 'Tortuosity', 'Lesion_Size', 'Image_Age', 'AMD_Risk_Score'
]
target = 'Diabetes'

# Encode categorical features
for col in ['Sex', 'Ethnicity', 'Smoking_Status', 'Alcohol_Level', 'Physical_Activity', 'Medication', 'Family_History']:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Drop rows with missing values in selected columns
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample training data
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Save model
joblib.dump(model, 'diabetes_risk_model.pkl')

print('Classifier trained and saved as diabetes_risk_model.pkl')
# Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f'Accuracy: {acc:.3f}')
print(f'ROC AUC: {roc_auc:.3f}')
print('Confusion Matrix:')
print(cm)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
