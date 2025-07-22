import os, subprocess
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "benchmarks/perf_results.csv"

def run_perf_analyzer(model_name, shape, concurrency):
    cmd = [
        "perf_analyzer",
        "-m", model_name,
        "--shape", shape,
        "--percentile=95",
        "--concurrency-range", concurrency,
        "-f", RESULTS_CSV
    ]
    subprocess.run(cmd, check=True)

def plot_results():
    df = pd.read_csv(RESULTS_CSV)
    df["p95_latency_ms"] = df["p95 latency"] / 1000
    plt.plot(df["Concurrency"], df["p95_latency_ms"], label="Triton Inference")
    
    # Load PyTorch baseline
    with open("benchmarks/pytorch_baseline.txt") as f:
        torch_latency = float(f.read().split(":")[1].strip().replace(" ms", ""))
        plt.axhline(y=torch_latency, color="r", linestyle="--", label="PyTorch Baseline")

    plt.xlabel("Concurrency")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Concurrency")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmarks/plot_perf_vs_latency.png")
    plt.show()

if __name__ == "__main__":
    os.makedirs("benchmarks", exist_ok=True)
    run_perf_analyzer("simple_cnn", "input_tensor:1,3,224,224", "1:4:1")
    plot_results()
