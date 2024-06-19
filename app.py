from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('models/Saeed.h5')

# Preprocessing function
image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

def preprocess_image(img):
    img = img.resize((224, 224))  # Change the size to (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = image_gen.standardize(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        img_preprocessed = preprocess_image(img)
        prediction = model.predict(img_preprocessed)
        confidence = np.max(prediction) * 100  # Get the highest confidence
        class_label = np.argmax(prediction, axis=1)[0]  # Get the predicted class label

        return jsonify({"class_label": int(class_label), "confidence": float(confidence)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)




