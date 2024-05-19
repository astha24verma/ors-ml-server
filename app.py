import tensorflow
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.src.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
# model.summary()
# img = cv2.imread("./images/1636.jpg")
# img = cv2.resize(img, (224, 224))
# img = np.array(img)
# img.shape
# expand_img = np.expand_dims(img, axis=0)
# pre_img = preprocess_input(expand_img)
# pre_img.shape
# result = model.predict(pre_img).flatten()
# normalized = result / norm(result)
# normalized.shape

# for extracting features from images
def extract_features(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# extract_features("./images/1636.jpg", model)

filename = []
feature_list = []

for file in os.listdir("./images"):
    filename.append(os.path.join("./images", file))

for file in tqdm(filename):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('features.pkl', 'wb'))
pickle.dump(filename, open('filenames.pkl', 'wb'))
