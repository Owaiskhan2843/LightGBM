import flask
import io
import string
import time
import os
import cv2
import numpy as np
import pickle, joblib
from PIL import Image
from flask import Flask, jsonify, request
from prep_image_module import prep_img
from keras.models import load_model
import h5py
VGG_model = load_model('VGG_model.h5')
model = joblib.load('LIGHT_GBM.pkl')

# def prepare_image(img):
#     img = Image.open(io.BytesIO(img))
#     img = img.resize((224, 224))
#     img = np.array(img)
#     img = np.expand_dims(img, 0)
#     return img


def predict(img1):
    # img1 = cv2.imread(path, cv2.IMREAD_COLOR)
    SIZE = 256
    # img = Image.open(io.BytesIO(img))
    # img1 = prep_img(img)
    img1 = cv2.resize(img1, (SIZE, SIZE))
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    
    img1 = img1/255
    # plt.imshow(img1)
    input_img = np.expand_dims(img1, axis=0)
    input_img_feature=VGG_model.predict(input_img)
    input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction = model.predict(input_img_features)[0] 
    # prediction = le.inverse_transform([prediction]) 
    return 'Cataract' if int(prediction) ==0 else 'Normal'

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
	
	file = request.form.get('file')
	img_bytes = file
	img = prep_img(img_bytes)
	
	return jsonify(prediction=predict(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Cataract Detection Using DL'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
