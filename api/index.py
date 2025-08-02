from flask import Flask, request, jsonify
import requests
import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# Load credentials from environment variables
API_KEY = os.environ.get("IBM_API_KEY")
SCORING_ENDPOINT = os.environ.get("IBM_SCORING_ENDPOINT")

# Preprocessing transform
# This part is crucial and should match your model's training
transform = A.Compose([
    A.Resize(224, 224), # Example resize
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Example normalization
    ToTensorV2()
])

def get_iam_token():
    """Generates an IBM IAM access token from the API key."""
    # This is a placeholder; you need to get the actual function from your IBM setup
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = 'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=' + API_KEY
    try:
        response = requests.post("https://iam.cloud.ibm.com/identity/token", headers=headers, data=data)
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting IAM token: {e}")
        return None

def logits_to_prediction(logits: list) -> int:
    """
    Converts the two CORAL logits into a final class prediction.
    This logic matches your KneeNet.predict() method.
    """
    # Convert the list of logits from the API to a PyTorch tensor
    logits_tensor = torch.tensor(logits)
    
    # Apply the sigmoid function to get probabilities
    probabilities = torch.sigmoid(logits_tensor)
    
    # The predicted grade is the sum of probabilities > 0.5
    predicted_grade = torch.sum(probabilities > 0.5).item()
    
    return predicted_grade

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'no image provided'}), 400

    image_file = request.files['image']
    
    # 1. Preprocess Image
    try:
        image_bytes = image_file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        numpy_image = np.array(pil_image)
        # Apply the transform
        transformed_image = transform(image=numpy_image)['image']
        # Convert tensor to list for JSON serialization
        image_data = transformed_image.unsqueeze(0).tolist()
    except Exception as e:
        return jsonify({'error': f'image preprocessing failed: {e}'}), 500
    
    # 2. Prepare payload for the IBM Watson Machine Learning API
    payload = {
        "input_data": [{
            "fields": ["image_data"],
            "values": image_data
        }]
    }

    # 3. Get token and send request
    token = get_iam_token()
    if not token:
        return jsonify({'error': 'authentication failed'}), 500

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.post(SCORING_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        
        # Extract the logits from the nested JSON response
        logits = result['predictions'][0]['values'][0]
        # Convert logits to the final grade (0, 1, or 2)
        final_prediction = logits_to_prediction(logits)
        
        return jsonify({'predicted_grade': final_prediction})
    except requests.exceptions.HTTPError as errh:
        return jsonify({'error': 'prediction failed', 'details': f'Http Error: {errh.response.text}'}), 500
    except requests.exceptions.RequestException as err:
        return jsonify({'error': 'prediction failed', 'details': f'Request Error: {err}'}), 500
    except (KeyError, IndexError) as e:
        return jsonify({'error': 'invalid response from IBM Watson', 'details': f'JSON parsing failed: {e}'}), 500