# Save this file as app.py
# This is the optimized version using Flask.
import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
from urllib import request, error
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- App Configuration ---
app = Flask(__name__)
# The Flask-CORS extension handles Cross-Origin Resource Sharing.
CORS(app)

# --- Environment Credentials ---
# In production, set these variables in your deployment environment (e.g., Vercel, AWS).
API_KEY = os.environ.get("IBM_API_KEY")
SCORING_ENDPOINT = os.environ.get("IBM_SCORING_ENDPOINT")

# --- Pre-processing and Post-processing (NumPy replacement) ---

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Replicates the original transform using only Pillow and NumPy."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    resized_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
    img_np = np.array(resized_image, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_img = (img_np - mean) / std

    transposed_img = normalized_img.transpose((2, 0, 1))
    
    return np.expand_dims(transposed_img, axis=0)

def logits_to_prediction_numpy(logits: list) -> int:
    """Converts model logits to a final prediction using only NumPy."""
    logits_array = np.array(logits, dtype=np.float32)
    probabilities = 1 / (1 + np.exp(-logits_array)) # Sigmoid function
    predicted_grade = np.sum(probabilities > 0.5)
    return int(predicted_grade)

def get_iam_token():
    """Generates an IBM IAM access token using Python's standard library."""
    if not API_KEY:
        print("Error: IBM_API_KEY not found in environment variables.")
        return None
    
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey".encode('utf-8')
    
    req = request.Request(url, data=data, headers=headers, method='POST')
    try:
        with request.urlopen(req) as resp:
            if resp.status != 200:
                print(f"Error getting IAM token. Status: {resp.status}")
                return None
            response_body = json.loads(resp.read().decode('utf-8'))
            return response_body.get("access_token")
    except error.URLError as e:
        print(f"Error during IAM token request: {e}")
        return None

# --- API Endpoint ---

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify(status='ok'), 200

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided in the request.'}), 400

    image_file = request.files['image']

    try:
        image_bytes = image_file.read()
        pil_image = Image.open(BytesIO(image_bytes))
        input_array = preprocess_image(pil_image)
    except Exception as e:
        return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 500

    token = get_iam_token()
    if not token:
        return jsonify({'error': 'Authentication failed. Could not get IAM token.'}), 500

    payload = {"input_data": [{"values": input_array.tolist()}]}
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    req_data = json.dumps(payload).encode('utf-8')
    pred_req = request.Request(SCORING_ENDPOINT, data=req_data, headers=headers, method='POST')

    try:
        with request.urlopen(pred_req) as resp:
            resp_body = resp.read()
            if resp.status >= 400:
                error_details = resp_body.decode('utf-8', errors='ignore')
                raise error.URLError(f"Prediction failed with status {resp.status}. Details: {error_details}")

            result = json.loads(resp_body.decode('utf-8'))
            model_output_logits = result['predictions'][0]['values'][0]
            final_prediction = logits_to_prediction_numpy(model_output_logits)
            
            return jsonify({'predicted_grade': final_prediction})

    except error.URLError as e:
        return jsonify({'error': 'Prediction request to IBM failed', 'details': str(e)}), 500
    except (KeyError, IndexError) as e:
        return jsonify({'error': 'Could not parse response from IBM Watson', 'details': f'Parsing failed: {str(e)}'}), 500