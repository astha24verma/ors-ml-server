from flask import Flask, request, jsonify, send_from_directory
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from flask_cors import CORS
import os
import logging
import pandas as pd

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model and feature lists
try:
    feature_list = np.array(pickle.load(open('features.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except FileNotFoundError as e:
    logging.error(f"File not found: {e.filename}")
    feature_list = None
    filenames = None

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load the CSV file containing filename and link information
csv_file = 'images.csv'
if os.path.exists(csv_file):
    image_df = pd.read_csv(csv_file)
else:
    logging.error(f"CSV file not found: {csv_file}")
    image_df = None

# Define the route for image processing
@app.route('/process_image', methods=['POST'])
def process_image():
    if feature_list is None or filenames is None:
        return jsonify({'error': 'Feature list or filenames not found'}), 500

    logging.debug("Received request for image processing")
    # Get the image data from the request
    image_file = request.files['image']
    image_data = image_file.read()

    # Convert the image data to a NumPy array
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Preprocess the image and get the features
    img = cv2.resize(image, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized_result = result / norm(result)

    # Find the nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([normalized_result])

    # Generate and save the output images with original filenames
    output_images = []
    output_links = []
    for index, file in enumerate(indices[0][1:6]):
        temp_img = cv2.imread(filenames[file])
        output_filename = os.path.basename(filenames[file])  # Get the original filename
        # output_path = os.path.join('static', output_filename)
        # cv2.imwrite(output_path, cv2.resize(temp_img, (512, 512)))
        output_images.append(output_filename)

        if image_df is not None:
            matched_links = []
            for csv_filename, link in zip(image_df['filename'], image_df['link']):
                if csv_filename == output_filename:
                    matched_links.append(link)
            
            if matched_links:
                output_links.append(matched_links[0])  # Take the first matched link
            else:
                output_links.append("Link not available")
        else:
            output_links.append("Link not available")


    logging.debug(f"Output images: {output_images}")
    logging.debug(f"Output links: {output_links}")

    # Return the output image paths and links as a JSON response
    return jsonify({'output_images': output_images, 'output_links': output_links})

# Route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
