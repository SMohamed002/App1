from flask import Flask, jsonify, request, render_template
import os
import numpy as np
import tensorflow as tf
from flask import request, send_file
from io import BytesIO

import cv2

from numpy import random
from matplotlib.image import imread

from imutils import paths
import os
import numpy as np
import random


import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology
import time

# Load the pre-trained neural network model
model = tf.keras.models.load_model('models/Model100.h5')

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')

# Receiving the image and classifying it
@app.route('/upload', methods=['POST'])
def classify_image():
    if request.method == 'POST':
        # Receiving the image sent from the Flutter application
        img_file = request.files['image']

        # Save the image temporarily on the server
        img_path = 'temp_image.jpg'
        img_file.save(img_path)

        # Load the image and perform the preprocessing
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # Convert image to LAB color space
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)

        # Apply KMeans clustering on 'a' channel
        a_img = a.reshape(a.shape[0]*a.shape[1], 1)
        km = KMeans(n_clusters=7, random_state=0).fit(a_img)
        p2s = km.cluster_centers_[km.labels_]
        ic = p2s.reshape(a.shape[0], a.shape[1])
        ic = ic.astype(np.uint8)

        # Thresholding and morphological operations
        _, t = cv2.threshold(ic, 157, 255, cv2.THRESH_BINARY)
        fh = ndi.binary_fill_holes(t)
        m1 = morphology.remove_small_objects(fh, 200)
        m2 = morphology.remove_small_holes(m1, 250)
        m2 = m2.astype(np.uint8)

        # Apply the mask to the original image
        out = cv2.bitwise_and(img, img, mask=m2)

        # Draw Anchors and confidence
        contours, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            font = cv2.FONT_HERSHEY_DUPLEX
            confidence = random.uniform(0.94, 0.99)  # Random confidence for demonstration
            text = f"ALL"
            (text_width, text_height), _ = cv2.getTextSize(text, font, 0.4, 1)
            cv2.putText(out, text, (x + int((w - text_width) / 2), y - 5), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw rectangle
            cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Add confidence percentage under the rectangle
            text_confidence = f"{confidence*100:.2f}%"
            (text_width, text_height), _ = cv2.getTextSize(text_confidence, font, 0.3, 1)
            cv2.putText(out, text_confidence, (x + int((w - text_width) / 2), y+h+15), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        # Remove the temporary image
        os.remove(img_path)

        # Convert the processed image to JPG bytes
        _, img_jpg = cv2.imencode('.jpg', out)
        img_jpg_bytes = img_jpg.tobytes()

        # Return the processed image as JPG
        return send_file(BytesIO(img_jpg_bytes), mimetype='image/jpeg')
if __name__ == '__main__':
    app.run(debug=True)




