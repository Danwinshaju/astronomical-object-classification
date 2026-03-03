import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# ==========================
# CONFIGURATION
# ==========================
DATA_DIR = "./dataset_resized"
MODEL_SAVE_PATH = "./models/astro_model_new.pth"
MAPPING_SAVE_PATH = "./models/class_mapping.json"

BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
VAL_SPLIT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# TRANSFORMS
# ==========================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# CUSTOM LEAF DATASET
# ==========================
class LeafFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self._scan_leaf_folders()

    def _scan_leaf_folders(self):
        print("🔍 Scanning leaf folders...\n")
        leaf_dirs = []

        for root, dirs, files in os.walk(self.root_dir):
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]

            if image_files and len(subdirs) == 0:
                leaf_dirs.append(root)

        leaf_dirs = sorted(leaf_dirs)
        self.classes = [os.path.basename(d) for d in leaf_dirs]
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for leaf_dir in tqdm(leaf_dirs, desc="Processing Classes"):
            cls_name = os.path.basename(leaf_dir)
            for file in os.listdir(leaf_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(leaf_dir, file),
                        class_to_idx[cls_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================
# LOAD DATASET
# ==========================
dataset = LeafFolderDataset(DATA_DIR, transform=train_transform)

num_classes = len(dataset.classes)

print(f"\n📂 Fine-Grained Classes Detected ({num_classes}):")
print(dataset.classes)
print(f"\n📊 Total images found: {len(dataset)}")

# Save class mapping
os.makedirs("./models", exist_ok=True)
with open(MAPPING_SAVE_PATH, "w") as f:
    json.dump(dataset.classes, f, indent=4)

print("✅ Class mapping saved.")

# Train / Validation split
train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==========================
# MODEL
# ==========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Freeze feature extractor initially
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0

# ==========================
# TRAINING LOOP
# ==========================
for epoch in range(EPOCHS):

    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

    # ===== TRAIN =====
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in train_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_bar.set_postfix(
            loss=running_loss/total,
            acc=100 * correct/total
        )

    train_acc = correct / total

    # ===== VALIDATION =====
    model.eval()
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            val_bar.set_postfix(
                acc=100 * correct/total
            )

    val_acc = correct / total

    print(f"📊 Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved!")

print("\n🎉 Fine-Grained Training Complete!")