from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Load the Keras model
model = tf.keras.models.load_model('models/Saeed.h5')

# Define your class labels
class_labels = ['Benign', 'Early Pre-B', 'Healthy', 'Pre-B', 'Pro-B']

def predict_image(img_path, model, class_labels):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])  # Get the confidence (probability) of the predicted class

    # Get the predicted class label
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Check if the POST request has the file part
    if 'imageFile' not in request.files:
        return 'No file part'

    file = request.files['imageFile']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return 'No selected file'

    # Save the uploaded file to the uploads folder
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get prediction and confidence
        predicted_class, confidence = predict_image(filepath, model, class_labels)

        # Return the result
        result = {
            'class': predicted_class,
            'confidence': float(confidence)  # Convert confidence to float for JSON serialization
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)




