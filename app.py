from flask import Flask, request, jsonify  # Mengimpor Flask, request, dan jsonify dari flask
from keras.models import load_model  # Mengimpor load_model dari keras.models
import numpy as np  # Mengimpor numpy sebagai np
from PIL import Image  # Mengimpor Image dari PIL
import io  # Mengimpor io

app = Flask(__name__)  # Membuat instance Flask
model = load_model('model.h5')  # Memuat model dari file 'model.h5'

@app.route('/predict', methods=['POST'])  # Mengatur rute '/predict' untuk metode POST
def predict():  # Fungsi untuk melakukan prediksi
    print(request.files)  # Menampilkan file yang dikirim
    if not request.files or len(request.files) == 0:  # Memeriksa jika tidak ada file
        return jsonify({'error': 'No image file provided'}), 400  # Mengembalikan error jika tidak ada file gambar
    
    # Menggunakan file pertama dalam ImmutableMultiDict
    file = next(iter(request.files.values()))  # Mengambil file pertama
    if not file or not file.filename.endswith(('png', 'jpg', 'jpeg')):  # Memeriksa tipe file
        return jsonify({'error': 'Invalid file type, expected an image'}), 400  # Mengembalikan error jika tipe file tidak valid
    
    try:  # Mencoba
        image = Image.open(file.stream).convert('RGB')  # Mengkonversi gambar ke RGB untuk memastikan 3 channel
        image_array = np.array(image.resize((224, 224))) / 255.0  # Menyesuaikan gambar
        prediction = model.predict(np.expand_dims(image_array, axis=0))  # Melakukan prediksi menggunakan model
        
        # Mendapatkan indeks dengan nilai prediksi tertinggi
        predicted_class = np.argmax(prediction[0])
        
        # Mapping indeks ke nilai nominal uang
        nominal_mapping = {
            0: 1000,
            1: 10000,
            2: 100000,
            3: 2000,
            4: 20000,
            5: 5000,
            6: 50000,
            7: 75000
        }
        
        # Mendapatkan nilai nominal berdasarkan prediksi
        predicted_nominal = nominal_mapping[predicted_class]
        
        # Menambahkan format mata uang Rupiah
        formatted_prediction = f"Rp {predicted_nominal:,}"
        
        return jsonify({'Hasil Prediksi': formatted_prediction})
        
    except Exception as e:  # Jika terjadi error
        return jsonify({'error': str(e)}), 500  # Mengembalikan error

if __name__ == '__main__':  # Jika file ini dijalankan secara langsung
    app.run(port=5000)  # Menjalankan aplikasi pada port 5000
