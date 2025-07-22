import time, torch
import torchvision.models as models

model = models.resnet18().eval()
dummy_input = torch.randn(1, 3, 224, 224)

start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)
end = time.time()

latency_ms = (end - start) / 100 * 1000
with open("benchmarks/pytorch_baseline.txt", "w") as f:
    f.write(f"Avg PyTorch latency: {latency_ms:.2f} ms")
