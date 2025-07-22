import requests
import json
import argparse
import numpy as np

def run_triton_inference(model_name, server_url, input_data):
    # Build payload
    payload = {
        "inputs": [
            {
                "name": "input_tensor",           # Must match Triton config.pbtxt
                "shape": list(np.array(input_data).shape),
                "datatype": "FP32",
                "data": input_data
            }
        ]
    }

    endpoint = f"{server_url}/v2/models/{model_name}/infer"
    response = requests.post(endpoint, json=payload)

    if response.status_code == 200:
        outputs = response.json().get("outputs", [])
        for output in outputs:
            print("🔮 Prediction:", output["data"])
    else:
        print(f"❌ Triton responded with status code {response.status_code}")
        print("Response:", response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="my_model", help="Name of the model as defined in Triton")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000", help="Triton server URL")
    args = parser.parse_args()

    # Simulate image-like input: batch of one RGB image
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32).tolist()

    run_triton_inference(args.model_name, args.server_url, dummy_input)
