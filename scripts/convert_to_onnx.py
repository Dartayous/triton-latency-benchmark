import torch
import torch.nn as nn
import torchvision.models as models
import argparse

# Dummy CNN or load your trained model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def convert_to_onnx(model_path, output_path):
    # Load or instantiate model
    model = SimpleCNN()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_tensor'],
        output_names=['output_tensor'],
        dynamic_axes={'input_tensor': {0: 'batch_size'}, 'output_tensor': {0: 'batch_size'}},
        opset_version=11
    )

    print(f"✅ ONNX model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to PyTorch model (.pth)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save ONNX model")
    args = parser.parse_args()

    convert_to_onnx(args.model_path, args.output_path)
