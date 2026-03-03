import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "./models/astro_model_new.pth"
MAPPING_PATH = "./models/class_mapping.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# LOAD CLASS MAPPING
# ==========================
if not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError("class_mapping.json not found. Retrain model.")

with open(MAPPING_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

num_classes = len(CLASS_NAMES)

print(f"✅ Loaded {num_classes} fine-grained classes.")

# ==========================
# RECREATE MODEL ARCHITECTURE
# ==========================
model = models.efficientnet_b0(weights=None)

# Replace classifier dynamically
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

# ==========================
# LOAD TRAINED WEIGHTS
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("astro_model.pth not found.")

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("✅ Model loaded successfully.")

# ==========================
# IMAGE TRANSFORM (same as training)
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_image(pil_image):

    image = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = CLASS_NAMES[predicted.item()]
    confidence_score = float(confidence.item())

    return predicted_label, confidence_score