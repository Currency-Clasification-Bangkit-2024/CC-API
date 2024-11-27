from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# URL API yang telah Anda deploy di Cloud Run
API_URL = "https://moneypredictionapi-863244423296.asia-southeast2.run.app/predict"  # Gantilah dengan URL Cloud Run Anda

@app.route('/')
def index():
    # Menampilkan halaman utama dengan form upload
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
    # Memeriksa apakah ada file gambar yang dikirimkan
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if file and (file.filename.endswith('.jpg') or file.filename.endswith('.jpeg') or file.filename.endswith('.png')):
        # Kirim gambar ke API menggunakan requests
        try:
            files = {'file': (file.filename, file.stream, file.content_type)}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                return jsonify({'prediction': response.json()['Hasil Prediksi']})
            else:
                return jsonify({'error': 'Gagal mendapatkan hasil dari API'}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File tidak valid'}), 400

if __name__ == '__main__':
    app.run(debug=True)
