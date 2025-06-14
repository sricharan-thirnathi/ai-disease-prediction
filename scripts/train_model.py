import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
DATA_PATH = os.path.join('data', 'Disease_Symptom.csv')
MODELS_DIR = 'models'
TARGET_COLUMN = 'prognosis'

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print("Original DataFrame head:")
    print(df.head())
    print("\nOriginal DataFrame info:")
    df.info()

    # --- Initial Memory Optimization on loaded data ---
    # Convert prognosis column to 'category' type as early as possible
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype('category')

    # Convert all other columns (symptoms) to smaller integer type (int8 for 0/1)
    initial_symptom_columns = [col for col in df.columns if col != TARGET_COLUMN]
    if not initial_symptom_columns:
        print("Error: No symptom columns identified based on TARGET_COLUMN. Check your dataset structure.")
        exit()

    for col in initial_symptom_columns:
        df[col] = df[col].fillna(0).astype(np.int8) # Fill NaN with 0 before converting to int8

    print(f"\nDataFrame memory usage after initial type optimization: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found. Please ensure your dataset is in the 'data' folder.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading or initial optimization: {e}")
    exit()

# --- 2. Data Cleaning and Preprocessing ---

# --- Handling Single-Instance Diseases (Fix for ValueError from train_test_split) ---
print("\nChecking disease class counts before filtering:")
initial_class_counts = df[TARGET_COLUMN].value_counts()
print(initial_class_counts)

# Identify classes with only 1 member
single_instance_classes = initial_class_counts[initial_class_counts == 1].index.tolist()

if single_instance_classes:
    print(f"\nWarning: The following diseases have only one instance and will be removed to allow stratification: {single_instance_classes}")
    df = df[~df[TARGET_COLUMN].isin(single_instance_classes)].copy() # Use .copy()
    print(f"Removed {len(single_instance_classes)} rows due to single instance diseases.")
else:
    print("\nNo single-instance diseases found. Proceeding normally.")

# --- Define final symptom columns (features) and encode diseases (target) ---
# Define symptom_columns *after* all filtering and cleaning of the DataFrame 'df'
symptom_columns = [col for col in df.columns if col != TARGET_COLUMN]

# The LabelEncoder and disease_labels MUST be based on the DataFrame *after* filtering.
le = LabelEncoder()
df['Disease_Encoded'] = le.fit_transform(df[TARGET_COLUMN])

# Save the LabelEncoder's classes (which are the actual disease names that remain)
disease_labels = list(le.classes_) # These are the ONLY diseases the model will know about
with open(os.path.join(MODELS_DIR, 'disease_labels.pkl'), 'wb') as f:
    pickle.dump(disease_labels, f)
print(f"Disease labels saved to {MODELS_DIR}/disease_labels.pkl")
print(f"Number of final disease classes: {len(disease_labels)}") # Verify count
print(f"Disease mapping (ID to Name): {dict(zip(le.transform(le.classes_), le.classes_))}")


# Create a mapping from symptom name to its index (column position)
# This uses the FINAL symptom_columns list
symptom_to_index = {symptom: i for i, symptom in enumerate(symptom_columns)}
with open(os.path.join(MODELS_DIR, 'symptom_encoder.pkl'), 'wb') as f:
    pickle.dump(symptom_to_index, f)
print(f"Symptom encoder (column mapping) saved to {MODELS_DIR}/symptom_encoder.pkl")
print(f"Total unique symptoms: {len(symptom_to_index)}")


# --- 3. Split Data ---
X = df[symptom_columns]
y = df['Disease_Encoded']

# Check class counts again after filtering, to confirm stratify will work
final_class_counts_for_split = y.value_counts()
# Ensure all remaining classes have at least 2 members for stratified split
if (final_class_counts_for_split < 2).any():
    print("\nFATAL ERROR: After filtering, some classes still have < 2 members. Cannot perform stratified split.")
    print("Final class counts for splitting:", final_class_counts_for_split)
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData split: Train {len(X_train)} samples, Test {len(X_test)} samples.")
print(f"Unique classes in y_train: {len(np.unique(y_train))}")
print(f"Unique classes in y_test: {len(np.unique(y_test))}")


# --- 4. Train Model ---
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate Model ---
y_pred = model.predict(X_test)

print(f"\nEvaluating model...")
print(f"Actual unique classes in y_test: {len(np.unique(y_test))}")
print(f"Total disease labels known by model: {len(disease_labels)}")

# --- CRITICAL FIX FOR THE VALUEERROR HERE ---
# Map the encoded labels in y_test to their actual names
# Only include target names for the classes actually present in y_test
test_set_unique_encoded_labels = np.unique(y_test)
dynamic_target_names = [disease_labels[i] for i in test_set_unique_encoded_labels]

print("\nClassification Report:")
# Pass the dynamically created target_names
print(classification_report(y_test, y_pred, target_names=dynamic_target_names))

print("\nConfusion Matrix:")
# For confusion matrix, it's best to specify the labels argument with the *encoded* unique labels
# that are actually present in the test set.
print(confusion_matrix(y_test, y_pred, labels=test_set_unique_encoded_labels) )


# --- 6. Save Model ---
model_path = os.path.join(MODELS_DIR, 'disease_prediction_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved to {model_path}")

print("\nTraining script finished. Model and encoders are ready for the web app!")