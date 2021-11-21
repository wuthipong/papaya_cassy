
from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

def model_predict(img_path):
    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    model = tensorflow.keras.models.load_model('papaya_model_tf.h5')
    preds = ""
    prediction = model.predict(data)
    if np.argmax(prediction)>=0 and np.argmax(prediction)<1:
        preds = f"Medium"
    elif np.argmax(prediction)==1:
        preds = f"ripe"
    else :
        preds = f"unripe"

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/Predict', methods=['POST'])
def upload():
        # Get the file from post request
        f = request.files['file']
        print(f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path)
        return """
        <!DOCTYPE html>
<html>
<head>
<title>Page Title</title>
</head>
<style>
    .bg-color {
        background-color: #B8DFD8;
    }
    .div-1 {
    position: absolute;
    width: 1350px;
    height: 600px;
    left: 94px;
    top: 62px;
    background: #E8F6EF;
    border-radius: 30px;
    }
    .fontstyle{
    font-family: Tajawal;
    font-style: normal;
    font-weight: bold;
    letter-spacing: -0.02em;
    color: #FFB319;
    margin-top: 20px; margin-left: 50px;
    }
    .line{
        margin-top: 20px; margin-left: 50px;
        background-color: #FFB319;
        width: 500px;
        height: 10px;
    }
    .box {
  width: 400px;
  height: 200px;
  padding: 50px;
  border: 4px solid #FFB319;
  border-radius: 20px;
}
</style>

<body class="bg-color">
    <div class="flex-container">
        <div class="div-1">
            <div style="float: left">
                <div class="fontstyle" style="font-size: 120px;">Check Your</div>
                <div class="fontstyle" style="font-size: 140px; margin-top: -10px;">Papaya!</div>
                <div class="line"></div>
                <div class="fontstyle" style="font-size: 30px;">use this website to predict <br/>
                                        your papaya by upload images <br/>
                                         that you want to predict its ripness.</div>
        </div>
        <div class="box" style="margin-left:55%; margin-top: 10%;">
            """+preds+"""
        </div>
    </div>
</div>

</body>
</html>
        """
    


if __name__ == '__main__':
    app.run(debug=True)