from flask import Flask, jsonify, request, render_template
import numpy as np
import tensorflow as tf


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

        # Load and classify the image using the model
        test_image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        # Classify the image using the model
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions)

        # Convert the results into known class names
        classes = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']
        result = classes[predicted_class]

        # Remove the temporary image
        os.remove(img_path)

        # Return the classification result as JSON
        return result


if __name__ == '__main__':
    app.run(debug=True)
    






