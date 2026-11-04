import numpy as np

TENSOR1_DIR = "/workspace/conv2d_silu_dev/test/build/20261104012134/0/yolov8s_subgraph_base/1/0"
TENSOR1_NAME = "Conv2D._model_0_conv_Conv.2.45.1793755295123294.input.2.npy"
TENSOR2_DIR = "/workspace/conv2d_silu_dev/test/build/20261104012134/0/yolov8s_subgraph_fuse/2/0"
TENSOR2_NAME = "Conv2dSilu.Conv2dSilu_00.2.43.1793755295137432.input.2.npy"


tensor1_path = f"{TENSOR1_DIR}/{TENSOR1_NAME}"
tensor2_path = f"{TENSOR2_DIR}/{TENSOR2_NAME}"
tensor1 = np.load(tensor1_path)
tensor2 = np.load(tensor2_path)
if tensor1.shape != tensor2.shape:
    print(f"shape不一致：{tensor1.shape} != {tensor2.shape}")
if tensor1.dtype != tensor2.dtype:
    print(f"dtype不一致：{tensor1.dtype} != {tensor2.dtype}")
if np.allclose(tensor1, tensor2):
    print("数据一致")
else:
    print("数据不一致!!!!!!!!")