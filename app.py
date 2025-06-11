from flask import Flask, request, render_template, send_file
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__)

# Initialize face analysis
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load swapper model
swapper = get_model("inswapper_128.onnx", download=True, providers=["CPUExecutionProvider"])

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/swap', methods=['POST'])
def swap():
    source = request.files['source']
    target = request.files['target']

    source_path = os.path.join(UPLOAD_FOLDER, 'source.jpg')
    target_path = os.path.join(UPLOAD_FOLDER, 'target.jpg')
    result_path = os.path.join(RESULT_FOLDER, 'result.jpg')

    source.save(source_path)
    target.save(target_path)

    # Read images
    src_img = cv2.imread(source_path)
    tgt_img = cv2.imread(target_path)

    # Get faces
    src_faces = face_app.get(src_img)
    tgt_faces = face_app.get(tgt_img)

    if len(src_faces) == 0 or len(tgt_faces) == 0:
        return "Face not detected in one of the images"

    # Perform swap on first detected face
    result = swapper.get(tgt_img, tgt_faces[0], src_faces[0].embedding)
    cv2.imwrite(result_path, result)

    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
