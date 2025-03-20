from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms, models

# Folders for uploaded files and extracted frames
UPLOAD_FOLDER = 'Uploaded_Files'
FRAME_FOLDER = 'Extracted_Frames'
STATIC_FOLDER = 'static'

# Ensure necessary directories exist
for folder in [UPLOAD_FOLDER, FRAME_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max upload size: 500MB

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False, device="cuda" if torch.cuda.is_available() else "cpu")

# Load the trained CNN model (EfficientNet-B3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Optimized Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def homepage():
    return render_template('home_page.html')

@app.route('/detection')
def detectionpage():
    return render_template('detection_page.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

def extract_faces_from_video(video_path, frame_interval=10, max_frames=5):
    """Extracts faces from a video and saves them as images."""
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_frame_folder = os.path.join(FRAME_FOLDER, video_name)
    os.makedirs(video_frame_folder, exist_ok=True)

    faces = []
    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or extracted_count >= max_frames:
            break  

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(frame_rgb)

            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob >= 0.90:
                        x1, y1, x2, y2 = map(int, box)
                        face = frame_rgb[y1:y2, x1:x2]

                        if face.size > 0:
                            face_filename = os.path.join(video_frame_folder, f"frame_{extracted_count}.jpg")
                            cv2.imwrite(face_filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                            faces.append(face_filename)
                            extracted_count += 1
        
        frame_count += 1

    cap.release()
    return faces, video_frame_folder

def extract_faces_from_image(image_path):
    """Extracts a face from an image and saves it."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    boxes, probs = mtcnn.detect(image_np)

    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        face = image_np[y1:y2, x1:x2]

        if face.size > 0:
            face_filename = os.path.join(FRAME_FOLDER, os.path.basename(image_path))
            cv2.imwrite(face_filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            return face_filename

    return None

def predict_fake_or_real(image_path):
    """Runs deepfake detection on an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)
            pred_value = torch.sigmoid(output).item()

        classification = "Real" if pred_value > 0.5 else "Fake"
        confidence = round(pred_value * 100 if classification == "Real" else (1 - pred_value) * 100)

        return classification, confidence

    except Exception as e:
        return "Error", f"Processing error: {str(e)}"

@app.route('/DetectImage', methods=['POST'])
def DetectImage():
    """Handles image uploads for deepfake detection."""
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded", "Score": "N/A", "Pred": "Error"})

    image = request.files['file']
    image_filename = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)

    try:
        image.save(image_filename)
        face_path = extract_faces_from_image(image_filename)

        if not face_path:
            return jsonify({"error": "No face detected", "Score": "N/A", "Pred": "No Face Found"})

        classification, confidence = predict_fake_or_real(face_path)

        if classification == "Error":
            return jsonify({"error": confidence, "Score": "N/A", "Pred": "Error"})

        return jsonify({"Score": confidence, "Pred": classification})
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}", "Score": "N/A", "Pred": "Error"})

@app.route('/Detect', methods=['POST'])
def DetectPage():
    """Handles video uploads for deepfake detection."""
    if 'file' not in request.files:
        return jsonify({"error": "No video uploaded", "Score": "N/A", "Pred": "Error"})

    video = request.files['file']
    video_filename = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)

    try:
        video.save(video_filename)
        if not os.path.exists(video_filename):
            return jsonify({"error": "File upload failed", "Score": "N/A", "Pred": "Error"})

        faces, frame_folder = extract_faces_from_video(video_filename)

        if len(faces) < 3:
            return jsonify({"error": "Not enough frames extracted (need at least 3)", "Score": "N/A", "Pred": "No Face Found"})

        frame_2_path = faces[2]  # Use only frame_2

        classification, confidence = predict_fake_or_real(frame_2_path)

        if classification == "Error":
            return jsonify({"error": confidence, "Score": "N/A", "Pred": "Error"})

        return jsonify({"Score": confidence, "Pred": classification, "Frame_Used": frame_2_path})

    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}", "Score": "N/A", "Pred": "Error"})

if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)

