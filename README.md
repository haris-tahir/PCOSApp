# PCOS Detection Web Application

This web application uses a trained deep learning model to detect Polycystic Ovary Syndrome (PCOS) from ultrasound images.

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model file `pcos_model.h5` in the root directory.

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click the upload button or drag and drop an ultrasound image (JPG or PNG)
2. Wait for the processing to complete
3. View the detection result

## Features

- Image upload and preview
- Real-time PCOS detection
- Clean, responsive UI with Tailwind CSS
- Loading indicator during processing

## Technical Details

- Backend: Flask
- Frontend: HTML, Tailwind CSS, JavaScript
- ML Framework: TensorFlow/Keras
- Image Processing: PIL, NumPy 