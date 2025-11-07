import os
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# ---- Config ----
DATA_DIR = r"C:\Users\Sai Tejus\HTF25-Team-232\data\helmet_classifier_data"  # <-- adjust path if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 32
OUT_PATH = "src/detection/models/helmet_classifier.pt"

# ---- Data transforms ----
train_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ---- Dataset and Loader ----
train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ---- Model: small pretrained backbone ----
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # 2 classes: helmet / no_helmet
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- Train loop ----
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | loss={epoch_loss:.4f} | acc={epoch_acc:.3f}")

# ---- Save model ----
torch.save(model.state_dict(), OUT_PATH)
print(f"✅ Saved helmet classifier to {OUT_PATH}")