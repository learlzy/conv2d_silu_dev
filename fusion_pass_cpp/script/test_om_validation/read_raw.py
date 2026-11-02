import numpy as np

def read_fp16_raw(file_path, elem_count=None):
    """
    读取fp16 raw二进制文件
    :param file_path: raw路径
    :param elem_count: 元素个数，如果给了就读固定数量；None=读全部
    :return: np.ndarray dtype=np.float16
    """
    with open(file_path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.float16)
    if elem_count is not None:
        arr = arr[:elem_count]
    return arr


if __name__ == "__main__":
    # ========== 在这里改你要读取的raw ==========
    raw_dir = "/mnt/workspace/conv2d_silu_dev/test/build/20261102095418/0/yolov8s_subgraph_base/1/0"
    raw_path = raw_dir + "/workspace.raw"
    # shape = [1, 32, 3, 3]
    # 如果你知道元素总数，可以填进去，不知道就写 None
    num_elem = None
    # ==========================================

    arr = read_fp16_raw(raw_path, num_elem)
    # arr = arr.reshape(shape)
    print(f"file: {raw_path}")
    print(f"total elements: {arr.size}")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print(arr)

