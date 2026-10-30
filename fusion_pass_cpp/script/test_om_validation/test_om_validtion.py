# compare_two_om.py
import acl
import numpy as np

# ===================== 配置区 =====================
OM_PATH_1 = "/workspace/conv2d_silu_dev/test/build/yolov8s_subgraph_base.om"
OM_PATH_2 = "/workspace/conv2d_silu_dev/test/build/yolov8s_subgraph_base.om"
DEVICE_ID = 0
INPUT_SHAPE = (8, 3, 640, 640)
INPUT_DTYPE = np.float32
SEED = 42
# 精度阈值
THRESH_ABS = 1e-5
THRESH_REL = 1e-5
# dump配置文件路径
DUMP_CONFIG_PATH = "./acl.json"
# =================================================

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

ACL_DTYPE_MAP = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.uint8,
    7: np.uint16,
    8: np.uint32,
    9: np.uint64,
    10: np.bool_,
}

def check_ret(ret_code, msg):
    if ret_code != 0:
        raise RuntimeError(f"[ACL_ERR] {msg}, ret={ret_code}")


class OmRunner:
    def __init__(self, om_path, dev_id):
        self.om_path = om_path
        self.dev_id = dev_id
        self.model_id = None
        self.model_desc = None
        self.ctx = None
        self.in_dataset = None
        self.out_dataset = None
        self.in_buffers = []
        self.out_buffers = []

    def init(self):
        # 【重点】不再在这里调用acl.init！全局只init一次
        ret = acl.rt.set_device(self.dev_id)
        check_ret(ret, "rt.set_device failed")
        self.ctx, ret = acl.rt.create_context(self.dev_id)
        check_ret(ret, "rt.create_context failed")

        self.model_id, ret = acl.mdl.load_from_file(self.om_path)
        check_ret(ret, f"load om {self.om_path} failed")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret(ret, "get_desc failed")

        in_num = acl.mdl.get_num_inputs(self.model_desc)
        out_num = acl.mdl.get_num_outputs(self.model_desc)
        print(f"[{self.om_path}] 输入数:{in_num}, 输出数:{out_num}")

    def _get_output_shape_dtype(self, idx):
        dims_info, ret = acl.mdl.get_output_dims(self.model_desc, idx)
        check_ret(ret, f"get_output_dims output[{idx}]")
        shape = tuple(dims_info["dims"])
        acl_dtype = acl.mdl.get_output_data_type(self.model_desc, idx)
        np_dtype = ACL_DTYPE_MAP.get(acl_dtype, np.uint8)
        return shape, np_dtype

    def _prepare_inputs(self, np_inputs: list[np.ndarray]):
        self.in_dataset = acl.mdl.create_dataset()
        self.in_buffers.clear()
        for arr in np_inputs:
            arr = np.ascontiguousarray(arr)
            byte_size = arr.nbytes
            dev_ptr, ret = acl.rt.malloc(byte_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret(ret, "rt.malloc input")
            arr_bytes = arr.tobytes()
            host_ptr = acl.util.bytes_to_ptr(arr_bytes)
            ret = acl.rt.memcpy(dev_ptr, byte_size, host_ptr, byte_size, ACL_MEMCPY_HOST_TO_DEVICE)
            check_ret(ret, "memcpy H2D")
            data_buf = acl.create_data_buffer(dev_ptr, byte_size)
            _, ret = acl.mdl.add_dataset_buffer(self.in_dataset, data_buf)
            check_ret(ret, "add input buffer")
            self.in_buffers.append((dev_ptr, byte_size))

    def _prepare_outputs(self):
        self.out_dataset = acl.mdl.create_dataset()
        self.out_buffers.clear()
        out_num = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(out_num):
            out_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            dev_ptr, ret = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret(ret, f"rt.malloc output[{i}]")
            data_buf = acl.create_data_buffer(dev_ptr, out_size)
            _, ret = acl.mdl.add_dataset_buffer(self.out_dataset, data_buf)
            check_ret(ret, "add output buffer")
            self.out_buffers.append((dev_ptr, out_size))

    def run(self, np_inputs: list[np.ndarray]) -> list[np.ndarray]:
        self._prepare_inputs(np_inputs)
        self._prepare_outputs()
        ret = acl.mdl.execute(self.model_id, self.in_dataset, self.out_dataset)
        check_ret(ret, "mdl.execute")

        out_list = []
        out_num = acl.mdl.get_num_outputs(self.model_desc)
        for idx, (dev_ptr, size) in enumerate(self.out_buffers):
            host_ptr, ret = acl.rt.malloc_host(size)
            check_ret(ret, "malloc_host")
            ret = acl.rt.memcpy(host_ptr, size, dev_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
            check_ret(ret, "memcpy D2H")
            raw_bytes = acl.util.ptr_to_bytes(host_ptr, size)
            out_shape, out_dtype = self._get_output_shape_dtype(idx)
            tensor = np.frombuffer(raw_bytes, dtype=out_dtype).reshape(out_shape)
            out_list.append(tensor)
            ret = acl.rt.free_host(host_ptr)
            check_ret(ret, "free_host")
        return out_list

    def release(self):
        in_num = acl.mdl.get_num_inputs(self.model_desc)
        if self.in_dataset is not None:
            for i in range(in_num):
                buf = acl.mdl.get_dataset_buffer(self.in_dataset, i)
                acl.destroy_data_buffer(buf)
                dev_ptr, _ = self.in_buffers[i]
                acl.rt.free(dev_ptr)
            acl.mdl.destroy_dataset(self.in_dataset)

        out_num = acl.mdl.get_num_outputs(self.model_desc)
        if self.out_dataset is not None:
            for i in range(out_num):
                buf = acl.mdl.get_dataset_buffer(self.out_dataset, i)
                acl.destroy_data_buffer(buf)
                dev_ptr, _ = self.out_buffers[i]
                acl.rt.free(dev_ptr)
            acl.mdl.destroy_dataset(self.out_dataset)

        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        if self.ctx is not None:
            acl.rt.destroy_context(self.ctx)
        print(f"[{self.om_path}] 模型资源释放完成\n")


def compare_array(a: np.ndarray, b: np.ndarray, out_idx: int):
    print(f"===== 输出[{out_idx}] 对比 =====")
    print(f"  A shape: {a.shape}, dtype:{a.dtype}")
    print(f"  B shape: {b.shape}, dtype:{b.dtype}")

    shape_ok = (a.shape == b.shape)
    dtype_ok = (a.dtype == b.dtype)
    if not shape_ok or not dtype_ok:
        print("  ❌ shape/dtype 不一致！")
        return False

    has_nan_a = np.any(np.isnan(a))
    has_nan_b = np.any(np.isnan(b))
    has_inf_a = np.any(np.isinf(a))
    has_inf_b = np.any(np.isinf(b))
    if has_nan_a or has_nan_b or has_inf_a or has_inf_b:
        print(f"  ⚠️  NaN/Inf 检测：A nan={has_nan_a}, inf={has_inf_a}; B nan={has_nan_b}, inf={has_inf_b}")

    all_eq = np.array_equal(a, b)
    abs_err = np.abs(a - b)
    max_abs_err = float(np.max(abs_err))

    denominator = np.maximum(np.abs(a), np.abs(b))
    mask = denominator > 1e-12
    rel_err = np.zeros_like(abs_err)
    rel_err[mask] = abs_err[mask] / denominator[mask]
    max_rel_err = float(np.max(rel_err))

    print(f"  全部精确相等: {all_eq}")
    print(f"  最大绝对误差: {max_abs_err:.8e}")
    print(f"  最大相对误差: {max_rel_err:.8e}")

    pass_flag = (max_abs_err < THRESH_ABS) and (max_rel_err < THRESH_REL)
    print(f"  ✅ 精度判定(abs<{THRESH_ABS}, rel<{THRESH_REL}): {pass_flag}\n")
    return pass_flag


if __name__ == "__main__":
    # ========== 全局只初始化一次acl ==========
    ret = acl.init(DUMP_CONFIG_PATH)
    check_ret(ret, "global acl.init failed")
    print("==== 全局ACL初始化完成 ====\n")

    # 固定种子，保证输入完全一模一样
    np.random.seed(SEED)
    input_tensor = np.random.rand(*INPUT_SHAPE).astype(INPUT_DTYPE)
    print(f"生成共用输入: shape={input_tensor.shape}, dtype={input_tensor.dtype}, seed={SEED}\n")

    runner1 = OmRunner(OM_PATH_1, DEVICE_ID)
    runner1.init()
    out1 = runner1.run([input_tensor])

    runner2 = OmRunner(OM_PATH_2, DEVICE_ID)
    runner2.init()
    out2 = runner2.run([input_tensor])

    # 输出数量对比
    if len(out1) != len(out2):
        print(f"❌ 输出数量不一致！om1:{len(out1)}, om2:{len(out2)}")
        total_pass = False
    else:
        total_pass = True
        for i in range(len(out1)):
            ok = compare_array(out1[i], out2[i], i)
            if not ok:
                total_pass = False
        print("==== 汇总结果 ====")
        if total_pass:
            print("✅ 所有输出精度通过！")
        else:
            print("❌ 存在输出精度不达标！")

    runner1.release()
    runner2.release()

    # ========== 全局最后一次finalize ==========
    acl.finalize()
    print("==== 全局ACL去初始化完成 ====")
