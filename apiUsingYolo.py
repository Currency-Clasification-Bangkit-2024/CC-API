from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
from keras.models import load_model
import cv2

app = Flask(__name__)

# Load YOLO model (format .h5)
MODEL_PATH = "modelYolo.h5"
model = load_model(MODEL_PATH)

# Nominal mapping for detected classes
nominal_mapping = {
    0: 1000,
    1: 2000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 50000,
    6: 100000
}

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload Gambar untuk Prediksi</title>
    <h1>Upload Gambar</h1>
    <form method=post enctype=multipart/form-data action="/predict">
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if not request.files or 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang disediakan'}), 400

    file = request.files['file']
    if not file.filename.endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Tipe file tidak valid, diharapkan gambar'}), 400

    try:
        # Membuka file gambar dan konversi ke RGB
        image = Image.open(file.stream).convert('RGB')
        image = np.array(image)

        # Preprocessing YOLO: Resize image dan normalisasi
        input_image = cv2.resize(image, (416, 416)) / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Prediksi menggunakan YOLO
        detections = model.predict(input_image)

        # Log hasil prediksi untuk debugging
        print("Detections Raw Output:", detections)

        # Validasi struktur output YOLO
        if len(detections) == 0 or len(detections[0]) == 0:
            return jsonify({'error': 'Tidak ada deteksi yang ditemukan'}), 200

        # Parse YOLO output
        results = []
        total_nominal = 0

        for detection in detections[0]:  # Iterasi hasil deteksi
            # Format YOLO: [x_center, y_center, width, height, confidence, class_probabilities...]
            confidence = detection[4]  # Confidence score
            if confidence > 0.5:  # Hanya deteksi dengan confidence > 0.5
                class_probabilities = detection[5:]  # Probabilitas setiap kelas
                class_id = np.argmax(class_probabilities)  # Indeks kelas dengan skor tertinggi
                nominal_value = nominal_mapping.get(class_id, None)
                if nominal_value:
                    total_nominal += nominal_value
                    results.append({
                        "nominal": nominal_value,
                        "confidence": round(confidence * 100, 2)
                    })

        # Jika tidak ada deteksi yang memenuhi threshold confidence
        if not results:
            return jsonify({'error': 'Tidak ada objek yang memenuhi confidence threshold'}), 200

        response = {
            "totalNominal": total_nominal,
            "detections": results
        }

        return jsonify(response)

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
