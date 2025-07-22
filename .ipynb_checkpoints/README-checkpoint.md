<p align="center">
  <img src="images/Triton_Deployemt_Banner.png" alt="Triton_Deployemt_Banner" width="100%">
</p>

# 🧠 Triton Latency Benchmark Suite  
**Optimizing PyTorch Inference with NVIDIA Triton + Performance Visualization**

Built by **Dartayous** — a Hollywood VFX veteran turned AI Engineer — this project benchmarks native PyTorch inference against NVIDIA Triton's production-grade deployment pipeline.

---

## 🚀 Highlights

- 📦 Deployed a PyTorch CNN (`simple_cnn`) using NVIDIA Triton
- ⚙️ Generated TensorRT engine and configured model repository
- 📈 Benchmarked with `perf_analyzer` (latency + throughput)
- 📊 Visualized performance metrics via `matplotlib`
- 🆚 Compared Triton vs raw PyTorch latency
- 🔁 CLI automation for repeatable benchmarking
- 🧹 Cleaned repo structure with `.gitignore` and size-optimized tracking
- 🌐 GitHub-ready: structured, documented, and deployable

---

## 📁 Project Structure

```text
ai_deployment_day17/
├── benchmarks/
│   ├── perf_results.csv           ← Triton benchmark output
│   ├── pytorch_baseline.txt       ← Native PyTorch latency baseline
│   └── plot_perf_vs_latency.png   ← Visualization of inference latency
├── model_repository/
│   └── simple_cnn/
│       ├── config.pbtxt           ← Triton model configuration
│       └── 1/
│           └── model.plan         ← TensorRT engine
├── scripts/
│   ├── measure_pytorch_baseline.py  ← Raw PyTorch benchmark runner
│   └── triton_latency_bench.py      ← Triton latency CLI tool + plotter
└── README.md


## 🧪 Quick Start
1️⃣ Benchmark PyTorch Inference
python scripts/measure_pytorch_baseline.py
Runs 100 inferences using ResNet18 and logs the average latency to pytorch_baseline.txt.


2️⃣ Launch Triton SDK Container
Run from your project root:
docker run --gpus all -it --rm --net=host ^
  -v ${PWD}:/workspace ^
  nvcr.io/nvidia/tritonserver:25.06-py3-sdk

Inside the container, install dependencies:
pip3 install matplotlib pandas


3️⃣ Run Triton Benchmark & Visualize Results
Inside the container:
python3 scripts/triton_latency_bench.py

This will:

Run perf_analyzer with concurrency levels 1–4

Save results to benchmarks/perf_results.csv

Generate a plot: plot_perf_vs_latency.png

Overlay raw PyTorch latency as a baseline


## 📊 Visualization Sample
🔴 Red dashed line: native PyTorch latency

🔵 Blue curve: Triton inference performance across concurrent loads

🧼 Git & Repository Notes
.gitignore includes:
__pycache__/
*.png
*.csv
*.txt
*.plan
.env/
venv/


Oversized files removed from repo:

cnn_model.pth (98 MB)

onnx_model.onnx (98 MB)

torch_cpu.dll (240 MB)

dnnl.lib (675 MB)

venv/ — excluded entirely

🔗 Users should download models separately or retrain locally.


## 🧰 Tech Stack
🐍 Python 3.x

🔬 PyTorch + torchvision

🚀 NVIDIA Triton Inference Server

⚡ TensorRT

🐳 Docker (SDK image: nvcr.io/nvidia/tritonserver:25.06-py3-sdk)

📈 CLI & visualization: perf_analyzer, matplotlib, pandas

## 🎬 Author
Dartayous
🎞️ Hollywood VFX professional turned AI Engineer
🧠 Specializing in deployment pipelines, performance optimization, and multimodal AI
🔧 Architect of this end-to-end benchmark suite


✨ Future Enhancements
Add argparse support to CLI tools

Timestamped benchmark logging

Web dashboard for real-time visualization

GitHub Actions: CI for benchmark validation

## 💡 About This Journey
This project evolved incrementally:

✅ Started with raw PyTorch inference

🚀 Deployed with Triton + TensorRT engine generation

🔍 Integrated SDK tools to measure latency and throughput

🔁 Added CLI automation + visualization

🧼 Cleaned commit history and optimized structure for public release