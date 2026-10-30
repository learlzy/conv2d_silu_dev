import numpy as np

DUMP_DIR = "/workspace/conv2d_silu_dev/test/build/20261030085435/0/yolov8s_subgraph_base/1/0"
DUMP_NAME = "Conv2D._model_0_conv_Conv.2.45.1793350476010194.input.0.npy"


DUMP_PATH = f"{DUMP_DIR}/{DUMP_NAME}"
arr = np.load(DUMP_PATH)
print(arr.shape, arr.dtype)