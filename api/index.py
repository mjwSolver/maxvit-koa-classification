# Save this file as app.py
import os
import requests
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- App Configuration ---
app = Flask(__name__)
# Enable CORS for all routes, allowing your React frontend to connect
CORS(app)

# --- Environment Credentials ---
API_KEY = os.environ.get("IBM_API_KEY")
SCORING_ENDPOINT = os.environ.get("IBM_SCORING_ENDPOINT")

# --- Preprocessing and Post-processing Functions ---

# FIX 1: Using the exact, correct preprocessing transform from your working script
transform = A.Compose([
    A.Resize(height=224, width=224, interpolation=cv2.INTER_LINEAR),
    # FIX 2: Using the correct ImageNet normalization values
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def get_iam_token():
    """Generates an IBM IAM access token from the API key."""
    if not API_KEY:
        print("Error: IBM_API_KEY not found in environment variables.")
        return None
    
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Error getting IAM token: {e}")
        return None

def logits_to_prediction(logits: list) -> int:
    """
    Converts the model's raw logits into a final predicted grade.
    This logic should match your model's specific output type (e.g., CORAL).
    """
    logits_tensor = torch.tensor(logits)
    probabilities = torch.sigmoid(logits_tensor)
    # The predicted grade is the sum of probabilities > 0.5
    predicted_grade = torch.sum(probabilities > 0.5).item()
    return int(predicted_grade)


# --- API Endpoint ---

# FIX 3: Added 'OPTIONS' to the list of allowed methods.
# This is the most likely fix for the `405 Method Not Allowed` CORS error.
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # The browser sends an OPTIONS request first (preflight) for CORS.
    # This block allows that request to succeed.
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided in the request.'}), 400

    image_file = request.files['image']

    try:
        image_bytes = image_file.read()
        
        # --- Preprocess the image from memory ---
        pil_image = Image.open(BytesIO(image_bytes)).convert("L") # Convert to Grayscale
        img_np = np.array(pil_image)
        # FIX 4: Convert grayscale back to 3-channel RGB for the model
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        transformed = transform(image=img_rgb)
        input_tensor = transformed["image"].unsqueeze(0) # Add batch dimension

    except Exception as e:
        return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 500

    token = get_iam_token()
    if not token:
        return jsonify({'error': 'Authentication failed. Could not get IAM token.'}), 500

    # FIX 5: Using the correct, simplified JSON payload structure
    payload = {
        "input_data": [{
            "values": input_tensor.tolist()
        }]
    }

    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.post(SCORING_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # --- Post-process the result to get a clean final grade ---
        model_output_logits = result['predictions'][0]['values'][0]
        final_prediction = logits_to_prediction(model_output_logits)
        
        return jsonify({'predicted_grade': final_prediction})

    except requests.exceptions.RequestException as e:
        error_details = response.text if 'response' in locals() else str(e)
        return jsonify({'error': 'Prediction request to IBM failed', 'details': error_details}), 500
    except (KeyError, IndexError) as e:
        return jsonify({'error': 'Could not parse response from IBM Watson', 'details': f'Parsing failed: {str(e)}'}), 500

# To run locally: `flask run` in your terminal
if __name__ == '__main__':
    app.run(debug=True, port=5001) # Use a different port like 5001 to avoid conflicts