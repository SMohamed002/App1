from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__)
app.static_folder = 'static'

# تحميل نموذج الشبكة العصبية المدربة سابقًا
script_dir = os.path.dirname(os.path.realpath(__file__))

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
        img_path = os.path.join(script_dir, 'temp_image.jpg')
        img_file.save(img_path)

        # تحميل النموذج داخل الدالة
        model_path = os.path.join(script_dir, 'models/Model100.h5')
        model = tf.keras.models.load_model(model_path)

        # تحميل الصورة وتصنيفها باستخدام النموذج
        test_image = image.load_img(img_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = preprocess_input(test_image)

        # تصنيف الصورة باستخدام النموذج
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions)

        # تحويل النتائج إلى الأسماء المعروفة
        classes = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']
        result = classes[predicted_class]

        # حذف الصورة المؤقتة
        os.remove(img_path)

        # عرض نتيجة التصنيف في الصفحة result.html
        return render_template('result.html', classification=result)

if __name__ == '__main__':
    app.run(debug=True)

