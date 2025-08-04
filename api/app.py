# Save this file as app.py
# This is the fully optimized version.
import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
from urllib import request, error
from bottle import Bottle, route, request, response, hook

app = Bottle()

# --- App Configuration & CORS ---
# This hook replaces the need for the Flask-CORS extension.
# It attaches the necessary headers to every response.
@hook('after_request')
def enable_cors():
    """Sets CORS headers for all outgoing responses."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, Authorization'

# --- Environment Credentials ---
# In production, set these variables in your deployment environment (e.g., Vercel, AWS).
API_KEY = os.environ.get("IBM_API_KEY")
SCORING_ENDPOINT = os.environ.get("IBM_SCORING_ENDPOINT")

# --- Pre-processing and Post-processing (NumPy replacement) ---

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Replicates the original transform using only Pillow and NumPy."""
    # Ensure the input image is 3-channel RGB.
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
        
    # 1. Resize the image (equivalent to A.Resize)
    resized_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # 2. Convert to NumPy array and normalize pixel values to [0, 1]
    img_np = np.array(resized_image, dtype=np.float32) / 255.0

    # 3. Apply ImageNet normalization (equivalent to A.Normalize)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_img = (img_np - mean) / std

    # 4. Transpose from (H, W, C) to (C, H, W) layout
    transposed_img = normalized_img.transpose((2, 0, 1))
    
    # 5. Add a batch dimension to match the model's expected input shape
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

@route('/api/predict', method=['POST', 'OPTIONS'])
def predict():
    # Bottle handles OPTIONS pre-flight requests automatically when a hook is present.
    if request.method == 'OPTIONS':
        return {}

    image_file = request.files.get('image')
    if not image_file:
        response.status = 400
        return json.dumps({'error': 'No image provided in the request.'})

    try:
        image_bytes = image_file.file.read()
        pil_image = Image.open(BytesIO(image_bytes))
        input_array = preprocess_image(pil_image)
    except Exception as e:
        response.status = 500
        return json.dumps({'error': f'Image preprocessing failed: {str(e)}'})

    token = get_iam_token()
    if not token:
        response.status = 500
        return json.dumps({'error': 'Authentication failed. Could not get IAM token.'})

    payload = {"input_data": [{"values": input_array.tolist()}]}
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    req_data = json.dumps(payload).encode('utf-8')
    pred_req = request.Request(SCORING_ENDPOINT, data=req_data, headers=headers, method='POST')

    try:
        with request.urlopen(pred_req) as resp:
            if resp.status >= 400:
                error_details = resp.read().decode('utf-8', errors='ignore')
                raise error.URLError(f"Prediction failed with status {resp.status}. Details: {error_details}")

            result = json.loads(resp.read().decode('utf-8'))
            model_output_logits = result['predictions'][0]['values'][0]
            final_prediction = logits_to_prediction_numpy(model_output_logits)
            
            response.content_type = 'application/json'
            return json.dumps({'predicted_grade': final_prediction})

    except error.URLError as e:
        response.status = 500
        return json.dumps({'error': 'Prediction request to IBM failed', 'details': str(e)})
    except (KeyError, IndexError) as e:
        response.status = 500
        return json.dumps({'error': 'Could not parse response from IBM Watson', 'details': f'Parsing failed: {str(e)}'})

# Note: The if __name__ == '__main__': block is removed
# as this file is intended for a serverless deployment environment.