from flask import Flask, request, jsonify , render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# تحميل نموذج الشبكة العصبية المدربة سابقًا
model = tf.keras.models.load_model('models/Model100.h5')

app = Flask(__name__)
app.static_folder = 'static'
@app.route('/')
def home():
    return render_template('index.html')

# استقبال الصورة وتصنيفها
@app.route('/classify', methods=['POST'])
def classify_image():
    if request.method == 'POST':
        # استقبال الصورة المُرسلة من التطبيق Flutter
        img_file = request.files['image']

        # حفظ الصورة مؤقتًا على الخادم
        img_path = 'temp_image.jpg'
        img_file.save(img_path)

        # تحميل الصورة وتصنيفها باستخدام النموذج
        test_image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        # تصنيف الصورة باستخدام النموذج
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions)

        # تحويل النتائج إلى الأسماء المعروفة
        classes = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']
        result = classes[predicted_class]

        # حذف الصورة المؤقتة
        os.remove(img_path)

        # إرجاع النتيجة كـ JSON
        return jsonify({'classification': result})


if __name__ == '__main__':
    app.run(debug=True)
