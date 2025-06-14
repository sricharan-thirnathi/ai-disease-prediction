import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# --- Configuration ---
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'disease_prediction_model.pkl')
SYMPTOM_ENCODER_PATH = os.path.join(MODELS_DIR, 'symptom_encoder.pkl')
DISEASE_LABELS_PATH = os.path.join(MODELS_DIR, 'disease_labels.pkl')

# --- Streamlit UI: set_page_config MUST BE THE FIRST EXECUTED Streamlit COMMANDS IN YOUR SCRIPT ---
st.set_page_config(
    page_title="AI Symptom Disease Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Encoders ---
@st.cache_resource # Cache the model loading for better performance
def load_resources():
    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
        symptom_to_index = pickle.load(open(SYMPTOM_ENCODER_PATH, 'rb'))
        disease_labels = pickle.load(open(DISEASE_LABELS_PATH, 'rb'))
        return model, symptom_to_index, disease_labels
    except FileNotFoundError:
        st.error(f"Error: Model or encoder files not found in '{MODELS_DIR}'. "
                 "Please ensure you've run 'scripts/train_model.py' successfully.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

# Call load_resources() AFTER st.set_page_config()
model, symptom_to_index, disease_labels = load_resources()

all_symptoms = sorted(list(symptom_to_index.keys())) # Get all possible symptoms from the encoder

# Get feature importances if the model has them (e.g., RandomForestClassifier)
# Map symptom names to their importance values
symptom_importances = {}
try:
    if hasattr(model, 'feature_importances_'):
        for symptom, index in symptom_to_index.items():
            symptom_importances[symptom] = model.feature_importances_[index]
    else:
        st.warning("Model does not have 'feature_importances_'. Feature importance plot will not be available.")
except Exception as e:
    st.warning(f"Could not retrieve feature importances: {e}")
    symptom_importances = {} # Reset to empty if error occurs

# --- Session State Initialization ---
if 'selected_symptoms_state' not in st.session_state:
    st.session_state.selected_symptoms_state = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


# --- Custom CSS for aesthetic improvements ---
st.markdown(
    """
    <style>
    /* General container padding for better spacing */
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    /* Style for the top prediction success box */
    .stAlert.stSuccess {
        border-left: 8px solid #4CAF50; /* Green border */
        background-color: #e6ffe6; /* Light green background */
        color: #333; /* Darker text */
    }
    /* Style for info boxes */
    .stAlert.stInfo {
        border-left: 8px solid #2196F3; /* Blue border */
        background-color: #e0f2ff; /* Light blue background */
        color: #333; /* Darker text */
    }
    /* Primary button styling */
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    /* Secondary button styling (for clear symptoms) */
    .stButton>button[kind="secondary"] {
        background-color: #f44336; /* Red button */
        color: white;
    }
    .stButton>button[kind="secondary"]:hover {
        background-color: #da190b;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar for Disclaimer and About ---
with st.sidebar:
    st.header("About This App")
    st.write(
        """
        This application uses a Machine Learning model (Random Forest Classifier) to predict potential diseases based on the symptoms you select.
        The model was trained on a dataset containing various symptoms and their corresponding disease prognoses.
        """
    )
    st.markdown("---")
    st.warning("### Important Medical Disclaimer:")
    st.write(
        """
        The predictions provided by this AI tool are for informational purposes only and are based on patterns learned from historical data.
        They **do not constitute medical advice, diagnosis, or treatment.**
        **Always consult a qualified healthcare professional** for accurate diagnosis and personalized medical guidance.
        Do not disregard professional medical advice or delay in seeking it because of something you have read here.
        """
    )
    st.markdown("---")
    st.caption("Developed with Streamlit and Scikit-learn for internship project.")

# --- Main Application UI ---
st.title("🩺 AI Symptom Disease Prediction")
st.markdown("---")

st.write(
    """
    Welcome! Select the symptoms you are currently experiencing from the list below, and our AI model will provide a prediction of potential diseases.
    """
)
st.markdown("---")

# --- Symptom Input ---
st.subheader("1. Select Your Symptoms")

selected_symptoms = st.multiselect(
    "Choose all symptoms you are experiencing:",
    options=all_symptoms,
    default=st.session_state.selected_symptoms_state, # Set default from session state
    help="Select symptoms from the dropdown. You can type to search."
)

# Update session state whenever multiselect changes
st.session_state.selected_symptoms_state = selected_symptoms


col1, col2 = st.columns([0.6, 0.4]) # Adjust column width for buttons

with col1:
    if st.button("Predict Disease", key="predict_button", type="primary"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom to get a prediction.")
        else:
            # --- Prepare Input for Model ---
            input_data = np.zeros(len(symptom_to_index)) # Initialize a zero array for all symptoms

            # Set 1 for selected symptoms
            for symptom in selected_symptoms:
                if symptom in symptom_to_index:
                    input_data[symptom_to_index[symptom]] = 1
                else:
                    st.warning(f"Symptom '{symptom}' not recognized by the model. It will be ignored.")

            input_data_reshaped = input_data.reshape(1, -1) # Reshape for a single prediction

            # --- Make Prediction ---
            # Get probabilities for all diseases
            prediction_proba = model.predict_proba(input_data_reshaped)[0]

            # Get top 3 predictions
            top_3_indices = np.argsort(prediction_proba)[::-1][:3]

            st.markdown("---")
            st.subheader("2. Prediction Results")

            # Store current prediction in history
            current_prediction_info = {
                "symptoms": selected_symptoms,
                "top_prediction": disease_labels[top_3_indices[0]],
                "confidence": prediction_proba[top_3_indices[0]] * 100,
                "full_results": []
            }

            # Display top predictions with confidence and optional visualization
            for i, idx in enumerate(top_3_indices):
                disease = disease_labels[idx]
                confidence = prediction_proba[idx] * 100 # Convert to percentage
                current_prediction_info["full_results"].append({"disease": disease, "confidence": confidence})

                if i == 0:
                    st.success(f"**Top Prediction:** **{disease}** (Confidence: {confidence:.2f}%)")
                    st.progress(confidence / 100)
                else:
                    st.info(f"**Possible Match {i+1}:** {disease} (Confidence: {confidence:.2f}%)")

            # Add to history
            st.session_state.prediction_history.append(current_prediction_info)

with col2:
    if st.button("Clear Symptoms", key="clear_button", type="secondary"):
        st.session_state.selected_symptoms_state = [] # Clear selected symptoms in session state
        st.rerun() # Rerun the app to clear the multiselect

st.markdown("---")

# --- New Feature: How it Works / Model Explanation ---
with st.expander("🔬 How It Works & Model Details"):
    st.write(
        """
        This application leverages a **Random Forest Classifier** model to predict diseases.
        Here's a brief overview:
        - **Data Input:** You provide symptoms (e.g., 'fever', 'cough') which are converted into a numerical format (1 if present, 0 if not).
        - **Model Training:** The Random Forest model was trained on a dataset where it learned patterns between various combinations of symptoms and their corresponding diseases.
        - **Prediction:** When you input symptoms, the model analyzes these patterns to predict the most probable diseases.
        - **Confidence:** The confidence percentage indicates how certain the model is about its prediction.
        """
    )
    st.subheader("Most Important Symptoms for Prediction")
    if symptom_importances:
        # Create a DataFrame for easy plotting
        importance_df = pd.DataFrame(symptom_importances.items(), columns=['Symptom', 'Importance'])
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10) # Get top 10

        st.bar_chart(importance_df.set_index('Symptom'))
        st.caption("These are the top 10 symptoms the model found most influential in making predictions across all diseases.")
    else:
        st.info("Feature importances could not be displayed (model type might not support it or an error occurred).")

# --- New Feature: Prediction History ---
if st.session_state.prediction_history:
    st.subheader("📊 Your Prediction History (This Session)")
    with st.expander("View Past Predictions"):
        for i, pred in enumerate(st.session_state.prediction_history):
            st.markdown(f"**Prediction {i+1}:**")
            st.write(f"**Symptoms Selected:** {', '.join(pred['symptoms'])}")
            st.write(f"**Top Result:** {pred['top_prediction']} (Confidence: {pred['confidence']:.2f}%)")
            # Optionally show full results
            # with st.expander("Details"):
            #     for res in pred['full_results']:
            #         st.write(f"- {res['disease']}: {res['confidence']:.2f}%")
            st.markdown("---")
else:
    st.subheader("📊 Your Prediction History (This Session)")
    st.info("No predictions yet in this session. Make a prediction to see history here!")


st.caption("Developed with Streamlit and Scikit-learn.")