from flask import Flask, request, jsonify, render_template
from model import predict_segmentation
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image and user inputs
        image_file = request.files['image']
        class_to_segment = request.form.get('class')
        time_range = request.form.get('time_range')

        # Save the uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)

        # Run the model
        output = predict_segmentation(image_array, class_to_segment, time_range)

        # Send response
        return jsonify({'status': 'success', 'segmentation_result': output.tolist()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
