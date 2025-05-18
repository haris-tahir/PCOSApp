import tensorflow as tf
import os

# Choose the path where your model is saved
# Either:
# 1. A directory: if saved with model.save("folder", save_format="tf")
# 2. A .h5 file: if saved with model.save("model.h5")

# Uncomment the one you have:
# model_input = "pcos_model_fixed"         # SavedModel folder
model_input = "pcos_model_fixed/pcos_model.h5"              # .h5 file

# Output file
tflite_output = "pcos_model.tflite"

# Load and convert
print(f"🔄 Loading model from: {model_input}")

if os.path.isdir(model_input):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_input)
else:
    model = tf.keras.models.load_model(model_input)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: enable optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save
tflite_model = converter.convert()
with open(tflite_output, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved as: {tflite_output}")
