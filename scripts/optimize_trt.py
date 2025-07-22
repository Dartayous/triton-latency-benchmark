import tensorrt as trt
import argparse

def build_engine(onnx_path, engine_path, use_fp16=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # 🧠 Explicit batch mode flag
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 🔍 Parse the ONNX model
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed.")

    # 🔧 Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 🎯 Define optimization profile for dynamic input shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name  # Should be 'input_tensor'
    profile.set_shape(input_name, (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
    config.add_optimization_profile(profile)

    # 🏗️ Build serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine.")

    # 💾 Save engine to disk
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ TensorRT engine saved to: {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT ONNX Optimizer")
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode")
    args = parser.parse_args()

    build_engine(args.onnx_path, args.output_path, args.fp16)


