# -----------------------------------------------------------------
# CW2: Phase 3 - Scoring Script (Based on Lab 4 - Deployment)
# -----------------------------------------------------------------
# This script defines the live API endpoint logic for scoring new data.

import json
import numpy as np
import pandas as pd
import os
import joblib

# The model will be loaded globally when the endpoint starts
model = None
# The vectorizer (our template dictionary) is also needed
vectorizer = None

def init():
    """
    This function is called when the container is initialized (only once).
    It loads the model and the vectorizer from the artifact path.
    """
    global model
    global vectorizer

    # Azure automatically maps the registered model files to a local path
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
    vectorizer_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'tfidf_vectorizer.joblib')

    # Load the trained Random Forest model and the vectorizer (from CW1)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("Initialization complete. Model and Vectorizer loaded.")

def run(raw_data):
    """
    This function is called for every incoming HTTP request (live log message).
    """
    try:
        # 1. Parse the incoming JSON data (simulating a live log line)
        data = json.loads(raw_data)['input_data']
        log_content = data[0] # Assume the raw log text is passed in

        # 2. FEATURE ENGINEERING (Reusing CW1 logic on the new log)
        # We must use the *saved* vectorizer to transform the new text
        new_template = pd.Series([log_content])
        
        # Transform the new log text into numerical features
        features = vectorizer.transform(new_template)
        
        # Convert the sparse result to a dense DataFrame for the model
        input_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())

        # 3. Predict
        prediction = model.predict(input_df)
        
        # 4. Format Output
        prediction_value = int(prediction[0])
        
        # 5. Guard-rail (Lab 9): Ensure the model returns a valid prediction
        if prediction_value not in [0, 1]:
            result = "Error: Invalid prediction value"
        else:
            result = {"is_anomaly": prediction_value, "prediction_text": "FAILURE DETECTED" if prediction_value == 1 else "Normal"}

        return json.dumps(result)

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})