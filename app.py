from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image = Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr = image_arr / 255.0  # Normalize the image
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension
    return image_arr

classes = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
model = load_model("models/NEW.h5")

@app.route('/')
def index():
    return render_template('index.html', appName="Intel Image Classification")

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'Error': str(e)})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        try:
            print("image loading....")
            image = request.files['fileup']
            print("image loaded....")
            image_arr = preprossing(image)
            print("predicting ...")
            result = model.predict(image_arr)
            print("predicted ...")
            ind = np.argmax(result)
            prediction = classes[ind]
            print(prediction)
            return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Intel Image Classification")
        except Exception as e:
            return render_template('index.html', prediction='Error: ' + str(e), appName="Intel Image Classification")
    else:
        return render_template('index.html', appName="Intel Image Classification")

if __name__ == '__main__':
    app.run(debug=True)
