import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image


MODEL_PATH = "models/image_detector.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create the same EfficientNet-B0 architecture
weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)


# Load our trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()


# Same preprocessing used by EfficientNet
transform = weights.transforms()


CLASS_NAMES = ["FAKE", "REAL"]


def predict_image(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_label = CLASS_NAMES[predicted_class.item()]

    confidence_value = confidence.item() * 100

    if predicted_label == "FAKE" and confidence_value >= 75:

        result = "YES — likely AI-generated"

    elif predicted_label == "REAL" and confidence_value >= 75:

     result = "NO — likely not AI-generated"

    else:

        result = "UNCERTAIN — insufficient evidence"

    return {
        "result": result,
        "label": predicted_label,
        "confidence": confidence_value
    }