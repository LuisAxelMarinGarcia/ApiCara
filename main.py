from flask import Flask, request, jsonify
import numpy as np
import cv2
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configuración del logger
handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
app.logger.addHandler(handler)

# Cargar el modelo entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained_model.xml')

# Constantes para la verificación
MI_ETIQUETA = 0  # Suponiendo que '0' es la etiqueta para tu rostro
UMBRAL_CONFIDENCIA = 50  # Define tu umbral de confianza

@app.route('/verify', methods=['POST'])
def verify():
    try:
        if 'image' not in request.files:
            app.logger.error('No image part in the request')
            return jsonify({'error': 'No image part in the request'}), 400
        
        file = request.files['image']
        if file.filename == '':
            app.logger.error('No image selected for uploading')
            return jsonify({'error': 'No image selected for uploading'}), 400

        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(roi_gray)
            if label == MI_ETIQUETA and confidence <= UMBRAL_CONFIDENCIA:
                return jsonify({'verified': True, 'confidence': confidence})

        return jsonify({'verified': False})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
