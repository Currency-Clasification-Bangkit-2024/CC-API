from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model secara lokal
MODEL_PATH = "modelYolo.pt"
model = YOLO(MODEL_PATH)

# Mapping nominal berdasarkan kelas
nominal_mapping = {
    0: "100ribu",
    1: "10ribu",
    2: "1ribu",
    3: "2ribu",
    4: "50ribu",
    5: "20ribu",
    6: "5ribu",
    7: "75ribu",
}


def process_image(image):
    """Proses gambar untuk mendeteksi objek."""
    results = model.predict(image, conf=0.4, iou=0.3)

    detections = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    detected_nominals = []
    total_value = 0

    for i, box in enumerate(detections):
        cls = int(classes[i])
        conf = confidences[i]

        if cls in nominal_mapping:
            nominal = nominal_mapping[cls]
            detected_nominals.append(nominal)
            total_value += int(nominal.replace("ribu", "")) * 1000

    total_value_formatted = f"{total_value // 1000}ribu"
    return detected_nominals, total_value_formatted


@app.route("/detect", methods=["POST"])
def detect():
    """API endpoint untuk mendeteksi nominal pada gambar."""
    if not request.files or "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang disediakan"}), 400

    file = request.files["image"]
    if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
        return jsonify({"error": "Tipe file tidak valid, diharapkan gambar"}), 400

    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        detected_nominals, total_value = process_image(image)

        response = {"detections": detected_nominals, "total_value": total_value}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Error memproses gambar: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
