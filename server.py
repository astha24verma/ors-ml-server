import cv2
from flask import Flask, request, jsonify, send_from_directory
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D, Input, Dense, Concatenate, Embedding, Flatten
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import pickle
import tensorflow as tf
from flask_cors import CORS
import os
import logging
import pandas as pd
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
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
# Load the pre-trained model for image processing
image_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_model.trainable = False
image_model = tf.keras.Sequential([
    image_model,
    GlobalMaxPooling2D()
])
# Load and preprocess the data
images_df = pd.read_csv('images.csv')
styles_df = pd.read_csv('styles.csv', on_bad_lines='skip')
# Remove '.jpg' from the 'filename' column in images_df
images_df['filename'] = images_df['filename'].str.replace('.jpg', '', regex=True)
# Convert the 'id' column to string type
styles_df['id'] = styles_df['id'].astype(str)
# Merge the dataframes based on the 'filename' and 'id' columns
merged_df = pd.merge(images_df, styles_df, left_on='filename', right_on='id')
print(merged_df.head())
# Create mappings for categorical variables
category_mapping = {category: index for index, category in enumerate(merged_df['subCategory'].unique())}
style_mapping = {style: index for index, style in enumerate(merged_df['usage'].unique())}
gender_mapping = {gender: index for index, gender in enumerate(merged_df['gender'].unique())}
color_mapping = {color: index for index, color in enumerate(merged_df['baseColour'].unique())}
# Convert categorical variables to numerical representations
merged_df['category'] = merged_df['subCategory'].map(category_mapping)
merged_df['style'] = merged_df['usage'].map(style_mapping)
merged_df['gender'] = merged_df['gender'].map(gender_mapping)
merged_df['color'] = merged_df['baseColour'].map(color_mapping)
#SCA Net architecture model
def create_model(num_categories, num_styles, num_genders, num_colors, num_recommendations):
    print('creating model')
    category_input = Input(shape=(1,))
    style_input = Input(shape=(1,))
    gender_input = Input(shape=(1,))
    color_input = Input(shape=(1,))
    category_embedding = Embedding(num_categories, 32)(category_input)
    style_embedding = Embedding(num_styles, 32)(style_input)
    gender_embedding = Embedding(num_genders, 16)(gender_input)
    color_embedding = Embedding(num_colors, 16)(color_input)
    category_flatten = Flatten()(category_embedding)
    style_flatten = Flatten()(style_embedding)
    gender_flatten = Flatten()(gender_embedding)
    color_flatten = Flatten()(color_embedding)
    concat = Concatenate()([category_flatten, style_flatten, gender_flatten, color_flatten])
    fc1 = Dense(256, activation='relu')(concat)
    fc2 = Dense(128, activation='relu')(fc1)
    output = Dense(num_recommendations, activation='sigmoid')(fc2)
    model = tf.keras.Model(inputs=[category_input, style_input, gender_input, color_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Create the categorical model
categorical_model = create_model(num_categories=len(category_mapping),
                                 num_styles=len(style_mapping),
                                 num_genders=len(gender_mapping),
                                 num_colors=len(color_mapping),
                                 num_recommendations=5)
def recommend_items(image, top_category, top_style, top_gender, top_color, num_recommendations=5):
    # Process the image using the image_model
    img = cv2.resize(image, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    image_features = image_model.predict(pre_img).flatten()
    bottom_categories = np.array([category_mapping['Bottomwear']], dtype=np.int32)
    shoe_categories = np.array([category_mapping['Shoes']], dtype=np.int32)
    # Check if top_style is None and replace it with a default value
    if top_style is None:
        top_style = 0  # or any other default value that makes sense in your context
    bottom_styles = np.array([top_style], dtype=np.int32)
    shoe_styles = np.array([top_style], dtype=np.int32)
    bottom_genders = np.array([top_gender], dtype=np.int32)
    shoe_genders = np.array([top_gender], dtype=np.int32)
    bottom_colors = np.array([top_color], dtype=np.int32)
    shoe_colors = np.array([top_color], dtype=np.int32)
    bottom_scores = categorical_model.predict([bottom_categories, bottom_styles, bottom_genders, bottom_colors])
    shoe_scores = categorical_model.predict([shoe_categories, shoe_styles, shoe_genders, shoe_colors])
    bottom_indices = np.argsort(bottom_scores, axis=1)[:, -num_recommendations:].flatten()[::-1]
    shoe_indices = np.argsort(shoe_scores, axis=1)[:, -num_recommendations:].flatten()[::-1]
    print(f"Bottom indices: {bottom_indices}")  # Added print statement
    print(f"Shoe indices: {shoe_indices}")  # Added print statement
    bottom_data = merged_df[(merged_df['category'] == category_mapping['Bottomwear']) &
                            (merged_df['gender'] == top_gender) &
                            (merged_df['color'] == top_color)]
    
    print(f"Number of NaN values in bottom_data: {bottom_data.isnull().sum().sum()}")
    
    valid_indices = np.clip(bottom_indices, 0, len(bottom_data) - 1)
    recommended_bottoms = bottom_data.iloc[valid_indices].reset_index(drop=True)
    print(f"Number of rows in bottom_data: {len(bottom_data)}")  # Added print statement
    print(f"Maximum index in bottom_indices: {np.max(bottom_indices)}")  # Added print statement

    shoe_data = merged_df[(merged_df['category'] == category_mapping['Shoes']) &
                          (merged_df['gender'] == top_gender) &
                          (merged_df['color'] == top_color)]
    recommended_shoes = shoe_data.iloc[shoe_indices].reset_index(drop=True)
    recommended_bottoms_filenames = recommended_bottoms['filename'].tolist()
    recommended_shoes_filenames = recommended_shoes['filename'].tolist()
    return recommended_bottoms_filenames, recommended_shoes_filenames, image_features
# Define the route for image processing
@app.route('/process_image', methods=['POST'])
def process_image():
    if feature_list is None or filenames is None:
        return jsonify({'error': 'Feature list or filenames not found'}), 500
    logging.debug("Received request for image processing")
    # Get the image data from the request
    image_file = request.files['image']
    image_data = image_file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Preprocess the image and get the features

    # Resize the image
    img = cv2.resize(image, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = image_model.predict(pre_img).flatten()
    normalized_result = result / norm(result)
    # Find the nearest neighbor
    neighbors = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([normalized_result])
    # Get the filename of the nearest neighbor image
    nearest_filename = os.path.basename(filenames[indices[0][1]])
    # Get the style, gender, and color values from the merged dataframe
    topwear_row = merged_df[merged_df['filename'] == nearest_filename.split('.')[0]]
    if not topwear_row.empty:
        top_style = topwear_row['style'].values[0]
        top_gender = topwear_row['gender'].values[0]
        top_color = topwear_row['color'].values[0]
        print(f"Topwear(nearest) Image: {nearest_filename} | Style: { top_style} | Gender: {top_gender} | Color: {top_color} ")
    else:
        # Handle the case when the style, gender, or color is not found
        top_style = None
        top_gender = None
        top_color = None
    # Get the top category from the merged dataframe
    top_category = merged_df[(merged_df['filename'] == nearest_filename.split('.')[0]) &
                            (merged_df['gender'] == top_gender) &
                            (merged_df['baseColour'] == merged_df.loc[merged_df['filename'] == nearest_filename.split('.')[0], 'baseColour'].values[0])]['category'].values[0]
    recommended_bottom_filenames, recommended_shoe_filenames, image_features = recommend_items(image, top_category, top_style, top_gender, top_color)

    # Get the link for the nearest image
    nearest_image_link = images_df[images_df['filename'] == nearest_filename.split('.')[0]]['link'].values[0]
    # Get the corresponding image links from images.csv
    bottom_links = images_df[images_df['filename'].isin(recommended_bottom_filenames)]['link'].tolist()
    shoe_links = images_df[images_df['filename'].isin(recommended_shoe_filenames)]['link'].tolist()
    response_data = {
        'top_link': nearest_image_link,
        'bottom_filenames': recommended_bottom_filenames,
        'bottom_links': bottom_links,
        'shoe_filenames': recommended_shoe_filenames,
        'shoe_links': shoe_links,
        'image_features': image_features.tolist()
    }
    return jsonify(response_data)
    
# Create an entry point for the function
if __name__ == '__main__':
    app.run(debug=True, port=5000)
