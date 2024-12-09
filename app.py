from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import os
from google.cloud import storage

app = Flask(__name__)

def download_model(bucket_name, model_file, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_file)
    blob.download_to_filename(local_path)
    print(f"Model {model_file} berhasil diunduh ke {local_path}")

BUCKET_NAME = "model_machine_learning_h5"
MODEL_FILE = "model1.h5"
LOCAL_MODEL_PATH = "/tmp/model1.h5"

if not os.path.exists(LOCAL_MODEL_PATH):
    download_model(BUCKET_NAME, MODEL_FILE, LOCAL_MODEL_PATH)

model = load_model(LOCAL_MODEL_PATH)

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
    if not request.files or len(request.files) == 0:
        return jsonify({'error': 'Tidak ada file gambar yang disediakan'}), 400

    file = next(iter(request.files.values()))
    if not file or not file.filename.endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Tipe file tidak valid, diharapkan gambar'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        image_array = np.array(image.resize((224, 224))) / 255.0
        prediction = model.predict(np.expand_dims(image_array, axis=0))

        confidence_scores = prediction[0]
        predicted_class = np.argmax(confidence_scores)
        confidence_score = confidence_scores[predicted_class] * 100  # Dalam persentase

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
        predicted_nominal = nominal_mapping[predicted_class]

        response = {
            "hasilPrediksi": f"{predicted_nominal}",  # Format 1.000
            "ConfidenceScore": float(round(confidence_score, 1))  # Konversi ke float
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
