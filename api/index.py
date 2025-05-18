import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

# Flask App Config
app = Flask(__name__, template_folder='../templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = '../uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TensorFlow Lite model
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'pcos_model.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully from:", model_path)
except Exception as e:
    print("❌ Failed to load TFLite model.")
    print(e)
    interpreter = None

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def process_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG or PNG'}), 400

    if interpreter is None:
        return jsonify({'error': 'TFLite model not loaded on server'}), 500

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_image = process_image(filepath)
        os.remove(filepath)

        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        confidence = float(prediction[0][0])
        result = confidence > 0.5

        return jsonify({
            'prediction': result,
            'probability': confidence,
            'message': 'PCOS Detected ✅' if result else 'PCOS Not Detected ❌'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Vercel compatibility
def handler(request):
    return app(request)

# Local test
if __name__ == '__main__':
    app.run(debug=True)
