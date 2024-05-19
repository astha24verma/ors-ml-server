import tensorflow
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.src.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors


feature_list = np.array(pickle.load(open('features.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = cv2.imread("./images/1611.jpg")
img = cv2.resize(img, (224, 224))
img = np.array(img)
expand_img = np.expand_dims(img, axis=0)
pre_img = preprocess_input(expand_img)
result = model.predict(pre_img).flatten()
normalized_result = result / norm(result)

neighbours = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
neighbours.fit(feature_list)

distance, indices = neighbours.kneighbors([normalized_result])
print(indices)

for index, file in enumerate(indices[0][1:6]):
    temp_img = cv2.imread(filenames[file])
    cv2.imwrite(f'output{index+1}.jpg', cv2.resize(temp_img, (512, 512)))
