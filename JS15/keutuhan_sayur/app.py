import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template_string
from skimage.color import rgb2hsv, hsv2rgb
from skimage import exposure

app = Flask(__name__)

# Load Model
MODEL_PATH = 'model_mobilenetv2_classifier.keras'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def preprocess_image(image_bytes):
    """
    Preprocess gambar untuk keutuhan sayur classifier.
    Pipeline sama dengan training: resize, brightness, contrast, histogram equalization, saturation.
    """
    try:
        # Decode gambar dari bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Gagal membaca gambar")
        
        # 1. Resize ke 224 × 224 piksel
        image = cv2.resize(image, (224, 224))
        
        # 2. BGR → RGB conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Normalisasi awal 0-1
        img = image.astype(np.float32) / 255.0
        
        # 4. Brightness & Contrast Adjustment
        img_tf = tf.constant(img)
        brightness_delta = 0.1
        img_tf = tf.image.adjust_brightness(img_tf, brightness_delta)
        contrast_factor = 1.3
        img_tf = tf.image.adjust_contrast(img_tf, contrast_factor)
        img = img_tf.numpy()
        img = np.clip(img, 0, 1)
        
        # 5. Histogram Equalization pada Kanal V (HSV)
        img_hsv = rgb2hsv(img)
        v_channel = img_hsv[:, :, 2]
        v_channel_eq = exposure.equalize_adapthist(v_channel)
        img_hsv[:, :, 2] = v_channel_eq
        img = hsv2rgb(img_hsv)
        
        # 6. Saturation Enhancement
        img_hsv_final = rgb2hsv(img)
        saturation_boost = 1.2
        img_hsv_final[:, :, 1] = np.clip(
            img_hsv_final[:, :, 1] * saturation_boost, 0, 1
        )
        img = hsv2rgb(img_hsv_final)
        
        # Final clip
        img = np.clip(img, 0, 1).astype(np.float32)
        
        # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
        input_data = np.expand_dims(img, axis=0)
        
        return input_data
        
    except Exception as e:
        raise ValueError(f"Error preprocessing gambar: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Keutuhan Sayur Classifier</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 0;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                text-align: center;
                max-width: 500px;
                width: 90%;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 32px;
            }
            p {
                color: #666;
                margin-bottom: 30px;
                font-size: 16px;
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            input[type="file"] {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                cursor: pointer;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                font-size: 18px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            button:hover {
                transform: scale(1.05);
            }
            .info {
                background: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
                color: #555;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🥬 Keutuhan Sayur Classifier</h1>
            <p>Klasifikasi sayur: Utuh atau Tidak Utuh</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">🔍 Analisis Gambar</button>
            </form>
            <div class="info">
                ℹ️ Upload gambar sayur (JPG, PNG, BMP)<br>
                Max 10MB
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.files or 'file' not in request.files:
            return render_template_string('''
            <div style="text-align:center; padding:50px; font-family: Arial;">
                <h2>❌ Error</h2>
                <p>Tidak ada file yang di-upload</p>
                <a href="/" style="color: #667eea;">← Kembali</a>
            </div>
            '''), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template_string('''
            <div style="text-align:center; padding:50px; font-family: Arial;">
                <h2>❌ Error</h2>
                <p>File belum dipilih</p>
                <a href="/" style="color: #667eea;">← Kembali</a>
            </div>
            '''), 400
        
        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return render_template_string('''
            <div style="text-align:center; padding:50px; font-family: Arial;">
                <h2>❌ Error</h2>
                <p>Tipe file tidak valid. Gunakan JPG, PNG, atau BMP</p>
                <a href="/" style="color: #667eea;">← Kembali</a>
            </div>
            '''), 400
        
        # Preprocess gambar
        image_data = preprocess_image(file.read())
        
        # Prediksi
        predictions = model.predict(image_data, verbose=0)
        probabilities = predictions[0]
        
        predicted_class_idx = np.argmax(probabilities)
        class_labels = ["🥬 Utuh", "🚫 Tidak Utuh"]
        confidence = float(probabilities[predicted_class_idx]) * 100
        
        utuh_prob = float(probabilities[0]) * 100
        tidak_utuh_prob = float(probabilities[1]) * 100
        
        prediction_label = class_labels[predicted_class_idx]
        
        # Color coding berdasarkan hasil
        color = "#27ae60" if predicted_class_idx == 0 else "#e74c3c"
        
        html_result = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hasil Prediksi</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 0;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                    text-align: center;
                    max-width: 500px;
                    width: 90%;
                }}
                .result {{
                    background: {color};
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin: 20px 0;
                    font-size: 28px;
                    font-weight: bold;
                }}
                .confidence {{
                    font-size: 20px;
                    margin: 20px 0;
                    color: #333;
                }}
                .probabilities {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .prob-item {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: 10px 0;
                    font-size: 16px;
                }}
                .prob-bar {{
                    width: 200px;
                    height: 20px;
                    background: #ddd;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                .prob-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    transition: transform 0.2s;
                }}
                a:hover {{
                    transform: scale(1.05);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Hasil Analisis</h1>
                <div class="result">{prediction_label}</div>
                <div class="confidence">
                    <strong>Confidence:</strong> {confidence:.2f}%
                </div>
                <div class="probabilities">
                    <h3>Probabilitas Kelas</h3>
                    <div class="prob-item">
                        <span>🥬 Utuh</span>
                        <div class="prob-bar">
                            <div class="prob-fill" style="width: {utuh_prob}%">
                                {utuh_prob:.1f}%
                            </div>
                        </div>
                    </div>
                    <div class="prob-item">
                        <span>🚫 Tidak Utuh</span>
                        <div class="prob-bar">
                            <div class="prob-fill" style="width: {tidak_utuh_prob}%">
                                {tidak_utuh_prob:.1f}%
                            </div>
                        </div>
                    </div>
                </div>
                <a href="/">← Analisis Lagi</a>
            </div>
        </body>
        </html>
        '''
        
        return html_result
        
    except ValueError as e:
        return render_template_string(f'''
        <div style="text-align:center; padding:50px; font-family: Arial;">
            <h2>❌ Error</h2>
            <p>{str(e)}</p>
            <a href="/" style="color: #667eea;">← Kembali</a>
        </div>
        '''), 400
    except Exception as e:
        return render_template_string(f'''
        <div style="text-align:center; padding:50px; font-family: Arial;">
            <h2>❌ Error</h2>
            <p>Terjadi kesalahan saat prediksi: {str(e)}</p>
            <a href="/" style="color: #667eea;">← Kembali</a>
        </div>
        '''), 500

if __name__ == '__main__':
    # Port 7860 untuk Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860, debug=False)
