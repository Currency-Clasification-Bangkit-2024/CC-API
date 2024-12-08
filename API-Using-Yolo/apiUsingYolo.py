from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO('modelYolo.pt')  # Pastikan modelYolo.pt sudah berada di direktori yang benar

# Mapping nominal berdasarkan kelas yang ada di model
nominal_mapping = {
    0: 100000,  # 100 ribu
    1: 10000,   # 10 ribu
    2: 1000,    # 1 ribu
    3: 20000,   # 20 ribu
    4: 2000,    # 2 ribu
    5: 50000,   # 50 ribu
    6: 5000,    # 5 ribu
    7: 75000    # 75 ribu
}

# HTML template for the upload form
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Currency Detection</title>
</head>
<body>
    <h1>Upload an Image to Detect Currency</h1>
    <form action="/detect" method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Upload Image</button>
    </form>

    {% if detections is not none %}
        <h2>Detection Results</h2>
        <p><strong>Detected Nominals:</strong> {{ detections }}</p>
        <p><strong>Total Value:</strong> {{ total_value }} IDR</p>
    {% endif %}

    {% if detection_info %}
        <h2>Detection Debug Info</h2>
        <ul>
        {% for info in detection_info %}
            <li>{{ info }}</li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

# Tambahkan variabel untuk menyimpan informasi deteksi
detection_info = []

def process_image(image):
    # Melakukan prediksi dengan threshold confidence dan NMS
    results = model.predict(image, conf=0.5, iou=0.4)  # Atur threshold untuk NMS dan confidence

    detections = results[0].boxes.xywh if results else []
    classes = results[0].boxes.cls if results else []
    confidences = results[0].boxes.conf if results else []

    detected_nominals = []
    total_value = 0
    detection_info.clear()

    # NMS secara internal sudah diterapkan oleh YOLOv8, jika perlu disesuaikan dapat dilakukan di sini
    for i, detection in enumerate(detections):
        cls = int(classes[i].item())
        conf = confidences[i].item()
        bbox = detection.cpu().numpy().tolist()

        # Tambahkan informasi deteksi untuk debugging
        detection_info.append(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {bbox}")

        # Filter deteksi dengan confidence rendah dan tambahkan verifikasi kelas
        if conf < 0.5:
            continue  # Abaikan deteksi dengan confidence < 0.5

        if cls in nominal_mapping:
            nominal = nominal_mapping[cls]
            detected_nominals.append(nominal)
            total_value += nominal

    print(f"Raw Output: {detection_info}")  # Debug log
    return detected_nominals, total_value, detection_info


# Contoh penggunaan dengan file gambar (misalnya dalam Flask route)
# Jika Anda menggunakan Flask untuk upload gambar, ini akan diproses dengan cara yang sama
def detect_from_file(image_path):
    image = cv2.imread(image_path)
    detected_nominals, total_value, detection_info = process_image(image)
    print(detected_nominals, total_value, detection_info)



@app.route('/')
def home():
    return render_template_string(html_template, detections=None, total_value=0)

@app.route('/detect', methods=['POST'])
def detect():
    if not request.files or len(request.files) == 0:
        return jsonify({'error': 'Tidak ada file gambar yang disediakan'}), 400

    # Get the first file from the uploaded files
    file = next(iter(request.files.values()))
    if not file or not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Tipe file tidak valid, diharapkan gambar'}), 400

    try:
        # Open and preprocess the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        detected_nominals, total_value, detection_info = process_image(image)

        # Prepare the response in JSON format
        response = {
            'detections': detected_nominals,  # List of detected nominal values
            'total_value': total_value,       # Total value of detected money
            'detection_info': detection_info  # Additional debug information
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error memproses gambar: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
