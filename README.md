# 🚀 Deploy on Hugging Face Spaces
---
title: Diabetes Lifestyle Suggestor
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
---

# Diabetes Lifestyle Suggestor

This project predicts diabetes risk and provides personalized lifestyle suggestions using a machine learning model and a Streamlit web app.

## Features
- Predicts diabetes risk score from patient data
- Gives actionable lifestyle suggestions based on risk and health factors
- Interactive Streamlit interface
- Data analysis and model evaluation scripts

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```sh
   python train_model.py
   ```
3. **Start the app:**
   ```sh
   streamlit run app.py
   ```

## Large Files
This repo uses [Git LFS](https://git-lfs.github.com/) for large files like `.csv` and `.pkl`.

## Files
- `app.py`: Streamlit app for prediction and suggestions
- `train_model.py`: Model training script
- `analyze_dataset.py`: Data analysis script
- `synthetic_amd_federated_dataset.csv`: Dataset
- `diabetes_risk_model.pkl`: Trained model (tracked with LFS)

## License
MIT
