from cv2 import data
from flask import Flask, render_template, request
import cv2

import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

# Model info
#model = tf.saved_model.load('saved_model.pb')
model = tf.keras.models.load_model('FruitDisease_plain.h5')
class_names = ['diseasedapple', 'diseasedbanana', 'diseasedorange', 'freshapples', 'freshbanana', 'freshoranges']

# home page
@app.route('/')
def home():
    return render_template('home.html')

# upload page
@app.route('/predict', methods=["POST"])
def predict():

    image = request.files['image']
    print(image)
    image.save("image.jpg")
    

    img = keras.preprocessing.image.load_img('image.jpg', target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = predictions[0]

    return render_template("predict.html", label = class_names[np.argmax(score)], score = 100 * np.max(score))


if __name__ == '__main__':
    app.run(debug=True) 