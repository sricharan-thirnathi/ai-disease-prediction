# AI Symptom Disease Prediction

This project aims to predict potential diseases based on a user's selected symptoms using a Machine Learning model. The web application is built with Streamlit, making it interactive and easy to use.

## Project Structure

*** READ IN CODE MODE FOR PROJECT STRUCTURE***

ai-disease-prediction/
├── data/
│   └── Disease_Symptom.csv      # raw dataset
├── models/
│   ├── disease_prediction_model.pkl  # Trained Random Forest Classifier
│   ├── symptom_encoder.pkl         # Mapping of symptom names to numerical features
│   └── disease_labels.pkl          # Mapping of numerical labels back to disease names
├── app/
│   └── app.py                     # Streamlit web application code
├── scripts/
│   └── train_model.py             # Script for data preprocessing and model training
├── .gitignore                     # Specifies intentionally untracked files to ignore
├── requirements.txt               # List of Python dependencies
└── README.md      



## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/ai-disease-prediction.git](https://github.com/YourGitHubUsername/ai-disease-prediction.git)
    cd ai-disease-prediction
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\Activate.ps1 (PowerShell) or .\venv\Scripts\activate.bat (Cmd)
    # On macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place your dataset:**
    * Ensure your `Disease_Symptom.csv` dataset is placed in the `data/` directory. The expected format is binary columns for symptoms (0 for absent, 1 for present) and a 'Disease' column for the target.
5.  **Train the model:**
    * This script will preprocess the data, train the model, and save the necessary `.pkl` files in the `models/` directory.
    ```bash
    python scripts/train_model.py
    ```
6.  **Run the Streamlit application:**
    ```bash
    streamlit run app/app.py
    ```
    This will open the application in your web browser, usually at `http://localhost:8501`.

## Disclaimer

This AI Symptom Disease Prediction tool is for informational purposes only and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns.
