import os
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Load your pre-trained model (replace 'model.h5' with your model's path)
model = load_model('braintumor.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize to match the model's input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the main page
@app.route('/')
def index():
    return render_template('home.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Process prediction (Assuming binary classification for simplicity)
    if prediction[0][0] > 0.5:
        result = 'Stroke Detected'
    else:
        result = 'No Stroke Detected'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
