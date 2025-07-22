import numpy as np
import tritonclient.http as httpclient

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")

# Create dummy input: shape [1, 3, 224, 224], dtype float32
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Define input tensor
input_tensor = httpclient.InferInput("input_tensor", dummy_input.shape, "FP32")
input_tensor.set_data_from_numpy(dummy_input)

# Define expected output
output_tensor = httpclient.InferRequestedOutput("output_tensor")

# Send inference request
response = client.infer(model_name="simple_cnn", inputs=[input_tensor], outputs=[output_tensor])

# Print output
print("Model Output:", response.as_numpy("output_tensor"))
