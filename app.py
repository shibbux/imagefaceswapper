!pip install -q insightface onnxruntime opencv-python gdown
!wget -q -O inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx


from google.colab import files
uploaded = files.upload()
# Make sure you upload files named exactly: source.jpg and target.jpg


import cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from google.colab.patches import cv2_imshow
from google.colab import files

# Initialize face detection
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

# Load the swapper model from local file
# Note: download=False since we already manually downloaded it
swapper = get_model("inswapper_128.onnx", download=False, providers=["CPUExecutionProvider"])

# Load images (already uploaded via Files tab)
src = cv2.imread("source.jpg")
tgt = cv2.imread("target.jpg")

# Detect faces
src_faces = app.get(src)
tgt_faces = app.get(tgt)

if not src_faces or not tgt_faces:
    print("😢 No face detected in one or both images.")
else:
    result = tgt.copy()
    for face in tgt_faces:
        result = swapper.get(result, face, src_faces[0], paste_back=True)

    cv2_imshow(result)
    cv2.imwrite("swapped.jpg", result)
    files.download("swapped.jpg")
    print("✅ Face-swapped image saved as 'swapped.jpg'")
