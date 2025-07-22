import torch
import torch.nn as nn
import torch.optim as optim
from scripts.cnn_model import SimpleCNN

def train_and_save(model_path="model/cnn_model.pth", num_classes=10, epochs=5):
    print("🔧 Initializing SimpleCNN...")
    model = SimpleCNN(num_classes=num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy input: one RGB image (3x224x224)
    inputs = torch.randn(1, 3, 224, 224)
    labels = torch.tensor([1])  # Fake label for training

    print("🎬 Starting training loop...")
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"🧠 Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # Save trained weights
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model weights saved to: {model_path}")

if __name__ == "__main__":
    train_and_save()
