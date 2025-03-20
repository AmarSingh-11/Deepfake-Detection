import torch
import torch.nn as nn
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from torchvision import models

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained CNN model (EfficientNet-B3)
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Define transformations to match your model's training process
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize MTCNN for face detection (used in existing functions)
mtcnn = MTCNN(keep_all=False, device=device)

def detect_deepfake(image_path):
    """
    Detects whether a given face in an image is real or fake using the trained CNN model.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Invalid image file"

    # Detect face using MTCNN (keeping existing functionality)
    boxes, _ = mtcnn.detect(img)
    
    if boxes is None:
        return "No face detected"

    # Extract the largest detected face (same logic as before)
    x1, y1, x2, y2 = map(int, boxes[0])
    face = img[y1:y2, x1:x2]

    if face.size == 0:
        return "Error: Face extraction failed"

    # Convert face to PIL Image for preprocessing
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    # Run the face through the model
    with torch.no_grad():
        output = model(face_tensor)
        prediction = torch.sigmoid(output).item()

    return "Fake" if prediction < 0.5 else "Real"
