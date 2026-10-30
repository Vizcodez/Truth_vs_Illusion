import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "data/cifake/train"
TEST_DIR = "data/cifake/test"
MODEL_DIR = "models"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001

# Start with a smaller dataset so the first training is manageable.
# Later, we can increase this for better performance.
MAX_TRAIN_PER_CLASS = 1000
MAX_TEST_PER_CLASS = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TRUTH VS. ILLUSION - IMAGE MODEL TRAINING")
print("=" * 60)
print(f"Device: {DEVICE}")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = weights.transforms()


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

train_dataset_full = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset_full = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print("Classes:", train_dataset_full.classes)
print("Class mapping:", train_dataset_full.class_to_idx)
print("Total training images:", len(train_dataset_full))
print("Total testing images:", len(test_dataset_full))


# ============================================================
# LIMIT DATASET FOR FIRST TRAINING
# ============================================================

def limit_per_class(dataset, max_per_class):
    selected_indices = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        selected_indices.extend(
            class_indices[:max_per_class]
        )

    return Subset(dataset, selected_indices)


train_dataset = limit_per_class(
    train_dataset_full,
    MAX_TRAIN_PER_CLASS
)

test_dataset = limit_per_class(
    test_dataset_full,
    MAX_TEST_PER_CLASS
)

print("\nImages used for this training:")
print("Training:", len(train_dataset))
print("Testing :", len(test_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CREATE EFFICIENTNET MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=weights
)

# Freeze the pretrained feature extractor initially
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace final classifier with binary classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(DEVICE)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "image_detector.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset_full.class_to_idx,
        "classes": train_dataset_full.classes
    },
    model_path
)

# Save class information separately
with open(
    os.path.join(MODEL_DIR, "image_classes.json"),
    "w"
) as f:

    json.dump(
        train_dataset_full.class_to_idx,
        f,
        indent=4
    )


print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Model saved to: {model_path}")
print("Classes:", train_dataset_full.classes)
print("=" * 60)
