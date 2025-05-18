import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="pcos_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Path to your test image (change this)
image_path = "test_images/sample.jpg"

input_data = preprocess_image(image_path)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

confidence = float(output_data[0][0])
print(f"✅ Prediction: {'PCOS Detected' if confidence > 0.5 else 'Not Detected'}")
print(f"🔍 Confidence: {confidence:.4f}")
