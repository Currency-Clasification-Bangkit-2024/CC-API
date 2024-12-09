from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('modelYolo.pt')

nominal_mapping = {
    0: 100000,
    1: 10000,
    2: 1000,
    3: 20000,
    4: 2000,
    5: 50000,
    6: 5000,
    7: 75000
}

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

detection_info = []

def process_image(image):
    results = model.predict(image, conf=0.5, iou=0.4)
    detections = results[0].boxes.xywh if results else []
    classes = results[0].boxes.cls if results else []
    confidences = results[0].boxes.conf if results else []

    detected_nominals = []
    total_value = 0
    detection_info.clear()

    for i, detection in enumerate(detections):
        cls = int(classes[i].item())
        conf = confidences[i].item()
        bbox = detection.cpu().numpy().tolist()

        detection_info.append(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {bbox}")

        if conf < 0.5:
            continue

        if cls in nominal_mapping:
            nominal = nominal_mapping[cls]
            detected_nominals.append(nominal)
            total_value += nominal

    print(f"Raw Output: {detection_info}")
    return detected_nominals, total_value, detection_info

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

    file = next(iter(request.files.values()))
    if not file or not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Tipe file tidak valid, diharapkan gambar'}), 400

    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        detected_nominals, total_value, detection_info = process_image(image)

        response = {
            'detections': detected_nominals,
            'total_value': total_value,
            'detection_info': detection_info
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error memproses gambar: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
