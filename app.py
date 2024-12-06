from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
from google.cloud import storage


# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model Keras yang sudah dilatih
# model = load_model('model.h5')

def download_model(bucket_name, model_file, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_file)
    blob.download_to_filename(local_path)
    print(f"Model {model_file} berhasil diunduh ke {local_path}")

BUCKET_NAME = "model_machine_learning_h5"
MODEL_FILE = "model.h5"
LOCAL_MODEL_PATH = "/tmp/model.h5"

if not os.path.exists(LOCAL_MODEL_PATH):
    download_model(BUCKET_NAME, MODEL_FILE, LOCAL_MODEL_PATH)

model = load_model(LOCAL_MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    # Memeriksa apakah ada file yang disertakan dalam permintaan
    if not request.files or len(request.files) == 0:
        return jsonify({'error': 'Tidak ada file gambar yang disediakan'}), 400

    # Mengambil file pertama dari permintaan
    file = next(iter(request.files.values()))
    # Memvalidasi tipe file
    if not file or not file.filename.endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Tipe file tidak valid, diharapkan gambar'}), 400

    try:
        # Membuka file gambar dan mengonversinya ke format RGB
        image = Image.open(file.stream).convert('RGB')
        # Mengubah ukuran gambar menjadi 224x224 dan menormalisasi nilai piksel
        image_array = np.array(image.resize((224, 224))) / 255.0
        # Membuat prediksi menggunakan model
        prediction = model.predict(np.expand_dims(image_array, axis=0))

        # Menentukan kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(prediction[0])

        # Memetakan kelas yang diprediksi ke nilai nominal
        nominal_mapping = {
            0: "1ribu",
            1: "10ribu",
            2: "100ribu",
            3: "2ribu",
            4: "20ribu",
            5: "5ribu",
            6: "50ribu",
            7: "75ribu"
        }

        # Mendapatkan nilai nominal untuk kelas yang diprediksi
        predicted_nominal = nominal_mapping[predicted_class]

        # Memformat hasil prediksi sebagai string mata uang
        formatted_prediction = f"Rp {predicted_nominal:,}"

        # Mengembalikan hasil prediksi sebagai respons JSON
        return jsonify({'Hasil Prediksi': formatted_prediction})

    except Exception as e:
        # Menangani setiap pengecualian yang terjadi selama pemrosesan
        return jsonify({'error': str(e)}), 500

# Menjalankan aplikasi Flask pada port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
