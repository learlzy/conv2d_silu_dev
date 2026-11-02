import numpy as np

DUMP_DIR = "/mnt/workspace/conv2d_silu_dev/test/build/20261102095418/0/yolov8s_subgraph_base/1/0"
DUMP_NAME = "Conv2D._model_0_conv_Conv.2.45.1793584458686230.output.0.npy"
SAVE_RAW = False  # True: 保存原始数据，False: 不保存


DUMP_PATH = f"{DUMP_DIR}/{DUMP_NAME}"
arr = np.load(DUMP_PATH)
print(arr.shape, arr.dtype)
print(arr)
if SAVE_RAW:
    raw_path = DUMP_PATH.replace(".npy", ".raw")
    arr.tofile(raw_path)
    print(f"导出完成：{raw_path}")