from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('modelYolo.pt')

nominal_mapping = {
    0: "100ribu",
    1: "10ribu",
    2: "1ribu",
    3: "20ribu",
    4: "2ribu",
    5: "50ribu",
    6: "5ribu",
    7: "75ribu"
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
        <p><strong>Total Value:</strong> {{ total_value }}</p>
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

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2p - x1p) * (y2p - y1p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def process_image(image):
    results = model.predict(image, conf=0.4, iou=0.3)

    detections = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    detected_nominals = []
    total_value = 0
    detection_info.clear()

    temp_detections = []
    for i, box in enumerate(detections):
        cls = int(classes[i])
        conf = confidences[i]
        bbox = box.tolist()

        detection_info.append(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {bbox}")

        if cls in nominal_mapping:
            nominal = nominal_mapping[cls]
            temp_detections.append({
                'nominal': nominal,
                'confidence': conf,
                'bbox': bbox
            })

    unique_detections = []
    for det in temp_detections:
        bbox = det['bbox']
        if not any(
            iou(bbox, u['bbox']) > 0.5 for u in unique_detections
        ):
            unique_detections.append(det)
            nominal = det['nominal']
            total_value += int(nominal.replace("ribu", "")) * 1000

    detected_nominals = [d['nominal'] for d in unique_detections]
    total_value_formatted = f"{total_value // 1000}ribu"
    return detected_nominals, total_value_formatted, detection_info

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
            'detections': detected_nominals,
            'total_value': total_value,
            'detection_info': detection_info
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error memproses gambar: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
