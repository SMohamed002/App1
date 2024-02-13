#importing all the required libraries
from flask import Flask,request,jsonify
import werkzeug
import json
import numpy as np
from keras.models import load_model
import keras.utils as image
import tensorflow as t
from keras.applications import InceptionV3

from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
#name of the trained model 
MODEL = 'models/Model100.h5'
#loading the model into API
model = load_model(MODEL)



#making numpy array of our labels 
classes= ['ALL','AML','CLL','CML','Healthy']



@app.route('/upload',methods=['POST'])
@cross_origin()
def upload():
    if(request.method=="POST"):
        #taking image from flutter front-end 
        imagefile=request.files['image']
        filename=werkzeug.utils.secure_filename(imagefile.filename)
        #saving image temporarily in "upload" folder 
        imagefile.save("./upload/"+filename)
        img="./upload/"+filename
        #the image is ready to get predicted
        ans=predict_image(img,model)
        #returning json object to our flutter front-end
        return jsonify(
            {
                "predictions":ans
            }
        )
    
#prediction of image using our trained model and labels text file
def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    
    prediction = model.predict(img_processed)
    
    index = np.argmax(prediction)
    #printing our prediction at the terminal.
    print("Prediction - {}".format(str(classes[index]).title()))
    #returning the prediction answer
    return str(classes[index])

if __name__ =="__main__":
    app.run(debug=True,port=5000,threaded=True)
