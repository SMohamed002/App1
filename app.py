from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import cv2
from sklearn.cluster import KMeans
from skimage import morphology
from scipy import ndimage as ndi
import base64

app = Flask(__name__)

# Load your trained model
model_path = 'models/Saeed.h5'  # ضع مسار النموذج الخاص بك هنا
model = load_model(model_path)

# Define your preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # تغيير حجم الصورة إلى 224x224
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def process_image(img_path, predicted_class_name):
    # قراءة الصورة
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # إذا كان التصنيف "Healthy"، نعيد الصورة الأصلية بدون تعديلات
    if predicted_class_name == "Healthy":
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    # تحويل الصورة إلى LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)

    # تطبيق KMeans
    a_reshaped = tf.reshape(a, [a.shape[0] * a.shape[1], 1])
    kmeans = KMeans(n_clusters=7, random_state=0).fit(a_reshaped)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = tf.reshape(clustered, [a.shape[0], a.shape[1]])
    clustered_img = tf.cast(clustered_img, tf.uint8)

    # تطبيق Threshold
    _, thresholded = cv2.threshold(clustered_img.numpy(), 141, 255, cv2.THRESH_BINARY)

    # ملء الفراغات وإزالة الأجسام الصغيرة
    filled_holes = ndi.binary_fill_holes(thresholded)
    cleaned = morphology.remove_small_objects(filled_holes, 200)
    cleaned = morphology.remove_small_holes(cleaned, 250)
    cleaned = tf.cast(cleaned, tf.uint8)

    # العثور على الكائنات ووضع Bounding Box
    contours, _ = cv2.findContours(cleaned.numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # رسم Bounding Box باللون الأحمر

    # تحويل الصورة إلى Base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)[0]

    # تحويل القيم العددية إلى نصوص
    class_names = ["Benign", "Early Pre-B", "Healthy", "Pre-B", "Pro-B"]
    predicted_class = tf.argmax(predictions).numpy()
    predicted_class_name = class_names[predicted_class]
    
    # دمج الأسماء مع القيم
    predictions_with_labels = [f"{class_names[i]}: {predictions[i]:.4f}" for i in range(len(predictions))]
    
    return predicted_class_name, predictions_with_labels

# Route for home page to upload image
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part', 400
    
    file = request.files['image']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)  # حفظ الملف في مجلد محلي
        file.save(file_path)
        
        predicted_class, predictions = predict_image(file_path)
        
        # Process the image
        processed_img_base64 = process_image(file_path, predicted_class)
        
        return jsonify({'predicted_class': predicted_class, 'predictions': predictions, 'processed_image': processed_img_base64})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # إنشاء مجلد لحفظ الملفات المرفوعة
    app.run(debug=True)




