from flask import Flask, request, jsonify
import requests
import os
# import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask_cors import CORS, cross_origin

# Create a Flask app instance and enable CORS
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load credentials from environment variables (e.g., from a .env file)
API_KEY = os.environ.get("IBM_API_KEY")
SCORING_ENDPOINT = os.environ.get("IBM_SCORING_ENDPOINT")

# Define the preprocessing transform
# This must match your model's training process exactly.
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

def get_iam_token():
    """Generates an IBM IAM access token from the API key."""
    if not API_KEY:
        print("IBM_API_KEY not found in environment variables.")
        return None
    
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = 'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=' + API_KEY
    try:
        response = requests.post("https://iam.cloud.ibm.com/identity/token", headers=headers, data=data)
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            print(f"Failed to get IAM token. Status: {response.status_code}, Response: {response.text}")
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
    
    return int(predicted_grade)

@app.route('predict', methods=['POST', 'OPTIONS'])
@cross_origin()  # Enable CORS for this route
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'no image provided'}), 400

    image_file = request.files['image']
    
    # 1. Preprocess the image
    try:
        image_bytes = image_file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        numpy_image = np.array(pil_image)
        transformed_image = transform(image=numpy_image)['image']
        # The tolist() method is essential to make the tensor serializable for the JSON payload.
        image_data = transformed_image.unsqueeze(0).tolist()
    except Exception as e:
        return jsonify({'error': f'image preprocessing failed: {str(e)}'}), 500
    
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
        response.raise_for_status() # Raise an exception for HTTP errors
        result = response.json()
        
        # 1. Extract the list of two logits from the API response
        model_output_logits = result['predictions'][0]['values'][0]
        
        # 2. Convert the logits to the final predicted grade
        final_prediction = logits_to_prediction(model_output_logits)
        
        # 3. Return the final, easy-to-understand result
        return jsonify({'predicted_grade': final_prediction})
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'prediction failed', 'details': str(e)}), 500
    except (KeyError, IndexError) as e:
        return jsonify({'error': 'invalid response from IBM Watson', 'details': f'JSON parsing failed: {str(e)}'}), 500

# Uncomment the following lines as it will be deployed on vercel
# if __name__ == '__main__':
#     app.run(debug=True)