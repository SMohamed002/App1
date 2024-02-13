from flask import Flask,request
import tensorflow as tf
import base64
import cv2
import numpy as np



new_model = tf.keras.models.load_model("models/Model100.h5")


app = Flask(__name__)



@app.route('/api',methods = ['Put'] )
def index():
       inputchar = request.get_data()
       imgdata = base64.b64decode(inputchar)
       filename = 'somthing.jpg'  
       with open(filename, 'wb') as f:
        f.write(imgdata)

       img = cv2.imread('somthing.jpg')
       resize = tf.image.resize(img, (224,224))
       yhat = new_model.predict(np.expand_dims(resize/255, 0))

       result = ""
       predicted_class = np.argmax(yhat)

        # Convert the results into known class names
        classes = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']
        result = classes[predicted_class]

        return result

        # Return the classification result as JSON
       
    


      



if __name__ == "__main__":
    app.run(debug=True)

