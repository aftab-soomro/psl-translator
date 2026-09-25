# PSL Translator

A machine learning project that recognizes Pakistan Sign Language (Urdu alphabet) hand signs. The pipeline extracts 21 hand landmarks (63 features) from images using MediaPipe, classifies them with classical ML models (KNN, SVM, Random Forest, MLP), and serves predictions through a Streamlit app with live webcam input.

## Folder structure

```
psl-translator/
├── data/
│   ├── raw/            # Original dataset images
│   ├── processed/      # Extracted landmarks (landmarks.csv)
│   └── custom_test/    # Custom recorded test images
├── notebooks/          # Exploration and experimentation notebooks
├── src/                # Core library code (config, landmarks, features, prediction)
├── models/             # Saved trained models (.joblib)
├── app/                # Streamlit application
│   └── pages/          # Additional Streamlit pages/features
├── reports/
│   └── figures/        # Generated graphs and figures for the report
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```
