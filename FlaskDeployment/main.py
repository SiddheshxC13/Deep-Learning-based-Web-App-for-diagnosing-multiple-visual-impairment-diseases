import os
from flask import Flask,render_template,request,jsonify
import tensorflow as tf
from util import base64_to_pil
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Image uploaded is converted from base64 to pillow
        img = base64_to_pil(request.json)
        # Image uploaded is converted too array
        img = image.img_to_array(img)
        # Image is normalized
        img = img / 255.0
        img = tf.expand_dims(img, 0)
        # Image is resized 
        img = tf.image.resize(img,(224,224))
        #Model is loaded
        model = tf.keras.models.load_model('models/EffiicientNet.h5',custom_objects = {'KerasLayer':hub.KerasLayer})
        #Prediction is made on images
        y_pred = model.predict(img)
        val = np.argmax(y_pred)
        result = ""
        mapp = {1: 'cat', 0: 'nor' , 2:'dia',3:'gla'}
        for k,v in mapp.items():
            if val==k:
                result = v
                break
        return jsonify(result=result)

if __name__ == '__main__':
    app.run(port=5002, threaded=False)