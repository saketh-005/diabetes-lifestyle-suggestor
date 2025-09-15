import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and dataset
model = joblib.load('diabetes_risk_model.pkl')
df = pd.read_csv('synthetic_amd_federated_dataset.csv')

st.title('Diabetes Risk Predictor & Lifestyle Suggestor')

st.sidebar.header('Patient Data Input')

# Example input fields

# All features used in training
feature_inputs = {}
feature_inputs['Age'] = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
feature_inputs['Sex'] = 0 if st.sidebar.selectbox('Sex', ['M', 'F']) == 'M' else 1
feature_inputs['Ethnicity'] = st.sidebar.selectbox('Ethnicity', df['Ethnicity'].unique())
feature_inputs['Smoking_Status'] = st.sidebar.selectbox('Smoking Status', df['Smoking_Status'].unique())
feature_inputs['Alcohol_Level'] = st.sidebar.selectbox('Alcohol Level', df['Alcohol_Level'].unique())
feature_inputs['BMI'] = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
feature_inputs['Physical_Activity'] = st.sidebar.selectbox('Physical Activity', df['Physical_Activity'].unique())
feature_inputs['Family_History'] = st.sidebar.selectbox('Family History', [0, 1])
feature_inputs['Hypertension'] = st.sidebar.selectbox('Hypertension', [0, 1])
feature_inputs['Cardio_Disease'] = st.sidebar.selectbox('Cardio Disease', [0, 1])
feature_inputs['Medication'] = st.sidebar.selectbox('Medication', df['Medication'].unique())
feature_inputs['HDL'] = st.sidebar.number_input('HDL', min_value=0.0, max_value=100.0, value=50.0)
feature_inputs['LDL'] = st.sidebar.number_input('LDL', min_value=0.0, max_value=300.0, value=100.0)
feature_inputs['Cholesterol'] = st.sidebar.number_input('Cholesterol', min_value=0.0, max_value=400.0, value=200.0)
feature_inputs['Triglycerides'] = st.sidebar.number_input('Triglycerides', min_value=0.0, max_value=500.0, value=150.0)
feature_inputs['HbA1c'] = st.sidebar.number_input('HbA1c', min_value=3.0, max_value=15.0, value=5.5)
feature_inputs['CMT'] = st.sidebar.number_input('CMT', min_value=0.0, max_value=500.0, value=200.0)
feature_inputs['RPE_Elevation'] = st.sidebar.number_input('RPE_Elevation', min_value=0.0, max_value=100.0, value=20.0)
feature_inputs['IRF'] = st.sidebar.number_input('IRF', min_value=0.0, max_value=10.0, value=0.0)
feature_inputs['SRF'] = st.sidebar.number_input('SRF', min_value=0.0, max_value=10.0, value=0.0)
feature_inputs['ILM_RPE'] = st.sidebar.number_input('ILM_RPE', min_value=0.0, max_value=500.0, value=200.0)
feature_inputs['Drusen_Volume'] = st.sidebar.number_input('Drusen_Volume', min_value=0.0, max_value=10.0, value=1.0)
feature_inputs['Drusen_Reflectivity'] = st.sidebar.number_input('Drusen_Reflectivity', min_value=0.0, max_value=1.0, value=0.5)
feature_inputs['EZ_Integrity'] = st.sidebar.number_input('EZ_Integrity', min_value=0.0, max_value=1.0, value=0.5)
feature_inputs['Choroidal_Thickness'] = st.sidebar.number_input('Choroidal_Thickness', min_value=0.0, max_value=500.0, value=200.0)
feature_inputs['HRF_Count'] = st.sidebar.number_input('HRF_Count', min_value=0, max_value=100, value=5)
feature_inputs['Texture'] = st.sidebar.number_input('Texture', min_value=0.0, max_value=1.0, value=0.5)
feature_inputs['Disc_Cup_Ratio'] = st.sidebar.number_input('Disc_Cup_Ratio', min_value=0.0, max_value=1.0, value=0.5)
feature_inputs['Tortuosity'] = st.sidebar.number_input('Tortuosity', min_value=0.0, max_value=1.0, value=0.5)
feature_inputs['Lesion_Size'] = st.sidebar.number_input('Lesion_Size', min_value=0.0, max_value=10.0, value=1.0)
feature_inputs['Image_Age'] = st.sidebar.number_input('Image_Age', min_value=0.0, max_value=100.0, value=50.0)
feature_inputs['AMD_Risk_Score'] = st.sidebar.number_input('AMD_Risk_Score', min_value=0.0, max_value=1.0, value=0.1)

# Encode categorical features as in training
input_data = {}
input_data['Age'] = [feature_inputs['Age']]
input_data['Sex'] = [feature_inputs['Sex']]
input_data['Ethnicity'] = [df['Ethnicity'].unique().tolist().index(feature_inputs['Ethnicity'])]
input_data['Smoking_Status'] = [df['Smoking_Status'].unique().tolist().index(feature_inputs['Smoking_Status'])]
input_data['Alcohol_Level'] = [df['Alcohol_Level'].unique().tolist().index(feature_inputs['Alcohol_Level'])]
input_data['BMI'] = [feature_inputs['BMI']]
input_data['Physical_Activity'] = [df['Physical_Activity'].unique().tolist().index(feature_inputs['Physical_Activity'])]
input_data['Family_History'] = [feature_inputs['Family_History']]
input_data['Hypertension'] = [feature_inputs['Hypertension']]
input_data['Cardio_Disease'] = [feature_inputs['Cardio_Disease']]
input_data['Medication'] = [df['Medication'].unique().tolist().index(feature_inputs['Medication'])]
input_data['HDL'] = [feature_inputs['HDL']]
input_data['LDL'] = [feature_inputs['LDL']]
input_data['Cholesterol'] = [feature_inputs['Cholesterol']]
input_data['Triglycerides'] = [feature_inputs['Triglycerides']]
input_data['HbA1c'] = [feature_inputs['HbA1c']]
input_data['CMT'] = [feature_inputs['CMT']]
input_data['RPE_Elevation'] = [feature_inputs['RPE_Elevation']]
input_data['IRF'] = [feature_inputs['IRF']]
input_data['SRF'] = [feature_inputs['SRF']]
input_data['ILM_RPE'] = [feature_inputs['ILM_RPE']]
input_data['Drusen_Volume'] = [feature_inputs['Drusen_Volume']]
input_data['Drusen_Reflectivity'] = [feature_inputs['Drusen_Reflectivity']]
input_data['EZ_Integrity'] = [feature_inputs['EZ_Integrity']]
input_data['Choroidal_Thickness'] = [feature_inputs['Choroidal_Thickness']]
input_data['HRF_Count'] = [feature_inputs['HRF_Count']]
input_data['Texture'] = [feature_inputs['Texture']]
input_data['Disc_Cup_Ratio'] = [feature_inputs['Disc_Cup_Ratio']]
input_data['Tortuosity'] = [feature_inputs['Tortuosity']]
input_data['Lesion_Size'] = [feature_inputs['Lesion_Size']]
input_data['Image_Age'] = [feature_inputs['Image_Age']]
input_data['AMD_Risk_Score'] = [feature_inputs['AMD_Risk_Score']]

input_df = pd.DataFrame(input_data)

if st.sidebar.button('Predict Risk'):
    risk_prob = model.predict_proba(input_df)[0][1]
    risk_percent = risk_prob * 100
    st.subheader(f'Predicted Diabetes Risk Score: {risk_percent:.1f}%')
    # Suggestions for every 10% risk increment
    suggestions = []
    bmi = input_df['BMI'][0]
    activity = input_df['Physical_Activity'][0]
    hba1c = input_df['HbA1c'][0]
    family_history = input_df['Family_History'][0]
    hypertension = input_df['Hypertension'][0]
    cholesterol = input_df['Cholesterol'][0]
    age = input_df['Age'][0]

    if risk_percent < 10:
        suggestions.append('Excellent! Maintain your healthy lifestyle and regular checkups.')
    elif risk_percent < 20:
        suggestions.append('Keep up the good work. Stay active and eat a balanced diet.')
    elif risk_percent < 30:
        suggestions.append('Continue healthy habits. Monitor your BMI and blood sugar.')
    elif risk_percent < 40:
        suggestions.append('Consider increasing physical activity and monitoring blood sugar.')
    elif risk_percent < 50:
        suggestions.append('Pay attention to your diet and exercise. Schedule regular screenings.')
    elif risk_percent < 60:
        suggestions.append('Adopt a more active lifestyle. Reduce sugar and salt intake.')
    elif risk_percent < 70:
        suggestions.append('Consult a healthcare provider for personalized advice. Monitor blood pressure and cholesterol.')
    elif risk_percent < 80:
        suggestions.append('Take steps to control risk factors: BMI, blood sugar, cholesterol, and blood pressure.')
    elif risk_percent < 90:
        suggestions.append('High risk: Seek medical advice. Consider lifestyle changes and regular monitoring.')
    else:
        suggestions.append('Very high risk: Immediate medical attention recommended. Follow strict lifestyle modifications.')

    # Add personalized suggestions based on input
    if bmi >= 25:
        suggestions.append('Aim for a BMI below 25 through healthy eating and regular exercise.')
    if activity == 0:
        suggestions.append('Increase physical activity. Try to get at least 150 minutes of moderate exercise per week.')
    if hba1c > 6.0:
        suggestions.append('Monitor and control your blood sugar. Consider consulting a dietitian.')
    if family_history == 1:
        suggestions.append('Since you have a family history, regular checkups are important.')
    if hypertension == 1:
        suggestions.append('Manage blood pressure with a low-salt diet and regular monitoring.')
    if cholesterol > 200:
        suggestions.append('Reduce cholesterol with a heart-healthy diet and physical activity.')
    if age > 50:
        suggestions.append('Schedule regular health screenings for diabetes and related conditions.')

    st.write('Lifestyle Suggestions:')
    for s in suggestions:
        st.write(f'- {s}')
