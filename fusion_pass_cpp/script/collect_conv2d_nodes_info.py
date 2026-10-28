import json
import csv
from collections import defaultdict

# ===================== 全局配置变量 =====================
INPUT_JSON_PATH = "/home/learlee/dev/projects/conv2d_silu_dev/test/build/yolov8s_om.json"
OUTPUT_CSV_PATH = "/home/learlee/dev/projects/conv2d_silu_dev/test/build/yolov8s_conv2d_info.csv"
# True: 收集x/filter/bias/y的实际shape/layout；False: 不收集，聚合不计入这些字段
COLLECT_TENSOR_INFO = True
# True: 收集tensor原始shape、原始origin_format，聚合key包含原始信息；False: 不收集，聚合不计入原始信息
COLLECT_TENSOR_ORIGIN_INFO = True
# ========================================================


def get_shape_from_desc(desc):
    """提取【实际】shape dim列表（desc顶层shape.dim）"""
    if "shape" not in desc or "dim" not in desc["shape"]:
        return []
    return desc["shape"]["dim"]


def get_real_layout(desc):
    """提取【实际】layout，desc顶层layout"""
    return desc.get("layout", "")


def parse_desc_attr(desc):
    """解析desc["attr"]数组，返回字典 {key:值}
    origin_shape -> list[int]
    origin_format -> str
    """
    attr_dict = {}
    attr_list = desc.get("attr", [])
    for item in attr_list:
        k = item.get("key")
        v_obj = item.get("value", {})
        if k == "origin_shape":
            # origin_shape: value -> list -> i
            lst = v_obj.get("list", {}).get("i", [])
            attr_dict["origin_shape"] = lst
        elif k == "origin_format":
            # origin_format: value -> s
            s_val = v_obj.get("s", "")
            attr_dict["origin_format"] = s_val
    return attr_dict


def main():
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    conv_records = []
    # 遍历graph列表
    for graph_item in graph_data.get("graph", []):
        ops = graph_item.get("op", [])
        for op in ops:
            if op.get("type") != "Conv2D":
                continue
            input_descs = op.get("input_desc", [])
            output_descs = op.get("output_desc", [])

            # ========= x 输入0 =========
            x_desc = input_descs[0] if len(input_descs) > 0 else {}
            x_attr = parse_desc_attr(x_desc)
            x_shape = get_shape_from_desc(x_desc)
            x_layout = get_real_layout(x_desc)
            x_origin_shape = x_attr.get("origin_shape", [])
            x_origin_layout = x_attr.get("origin_format", "")

            # ========= filter 输入1 =========
            filter_desc = input_descs[1] if len(input_descs) > 1 else {}
            filter_attr = parse_desc_attr(filter_desc)
            filter_shape = get_shape_from_desc(filter_desc)
            filter_layout = get_real_layout(filter_desc)
            filter_origin_shape = filter_attr.get("origin_shape", [])
            filter_origin_layout = filter_attr.get("origin_format", "")

            # ========= bias 输入2 =========
            bias_desc = input_descs[2] if len(input_descs) > 2 else {}
            bias_attr = parse_desc_attr(bias_desc)
            bias_shape = get_shape_from_desc(bias_desc)
            bias_layout = get_real_layout(bias_desc)
            bias_origin_shape = bias_attr.get("origin_shape", [])
            bias_origin_layout = bias_attr.get("origin_format", "")

            # ========= y输出0 =========
            y_desc = output_descs[0] if len(output_descs) > 0 else {}
            y_attr = parse_desc_attr(y_desc)
            y_shape = get_shape_from_desc(y_desc)
            y_layout = get_real_layout(y_desc)
            y_origin_shape = y_attr.get("origin_shape", [])
            y_origin_layout = y_attr.get("origin_format", "")

            # ========= conv属性 strides pads dilations =========
            attrs = op.get("attr", [])
            strides = []
            pads = []
            dilations = []
            for attr in attrs:
                k = attr.get("key")
                val = attr.get("value", {})
                if k == "strides":
                    strides = val.get("list", {}).get("i", [])
                elif k == "pads":
                    pads = val.get("list", {}).get("i", [])
                elif k == "dilations":
                    dilations = val.get("list", {}).get("i", [])

            # ========= 构造聚合key，开关控制是否加入 =========
            key_parts = []
            # 原始tensor信息
            if COLLECT_TENSOR_ORIGIN_INFO:
                key_parts.extend([
                    tuple(x_origin_shape),
                    x_origin_layout,
                    tuple(filter_origin_shape),
                    filter_origin_layout,
                    tuple(bias_origin_shape),
                    bias_origin_layout,
                    tuple(y_origin_shape),
                    y_origin_layout,
                ])
            # 实际tensor信息
            if COLLECT_TENSOR_INFO:
                key_parts.extend([
                    tuple(x_shape),
                    x_layout,
                    tuple(filter_shape),
                    filter_layout,
                    tuple(bias_shape),
                    bias_layout,
                    tuple(y_shape),
                    y_layout,
                ])
            # 卷积参数永远参与聚合
            key_parts.extend([str(strides), str(pads), str(dilations)])
            record_key = tuple(key_parts)

            conv_records.append((
                record_key,
                x_origin_shape, x_origin_layout,
                filter_origin_shape, filter_origin_layout,
                bias_origin_shape, bias_origin_layout,
                y_origin_shape, y_origin_layout,
                x_shape, x_layout,
                filter_shape, filter_layout,
                bias_shape, bias_layout,
                y_shape, y_layout,
                strides, pads, dilations
            ))

    # 计数、缓存样例
    counter = defaultdict(int)
    raw_store = dict()
    for item in conv_records:
        rk = item[0]
        counter[rk] += 1
        if rk not in raw_store:
            raw_store[rk] = item[1:]

    # csv表头，原始信息放最前面
    csv_header = [
        "x_origin_shape",
        "x_origin_layout",
        "filter_origin_shape",
        "filter_origin_layout",
        "bias_origin_shape",
        "bias_origin_layout",
        "y_origin_shape",
        "y_origin_layout",
        "x_shape",
        "x_layout",
        "filter_shape",
        "filter_layout",
        "bias_shape",
        "bias_layout",
        "y_shape",
        "y_layout",
        "strides",
        "pads",
        "dilations",
        "count",
    ]

    rows = []
    for key, cnt in counter.items():
        (
            x_origin_shape, x_origin_layout,
            filter_origin_shape, filter_origin_layout,
            bias_origin_shape, bias_origin_layout,
            y_origin_shape, y_origin_layout,
            x_shape, x_layout,
            filter_shape, filter_layout,
            bias_shape, bias_layout,
            y_shape, y_layout,
            strides, pads, dilations
        ) = raw_store[key]

        row = []
        # 原始tensor字段
        if COLLECT_TENSOR_ORIGIN_INFO:
            row.extend([
                x_origin_shape, x_origin_layout,
                filter_origin_shape, filter_origin_layout,
                bias_origin_shape, bias_origin_layout,
                y_origin_shape, y_origin_layout,
            ])
        else:
            row.extend(["", "", "", "", "", "", "", ""])

        # 实际tensor字段
        if COLLECT_TENSOR_INFO:
            row.extend([
                x_shape, x_layout,
                filter_shape, filter_layout,
                bias_shape, bias_layout,
                y_shape, y_layout,
            ])
        else:
            row.extend(["", "", "", "", "", "", "", ""])

        row.extend([str(strides), str(pads), str(dilations), cnt])
        rows.append(row)

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(csv_header)
        writer.writerows(rows)

    print(f"✅ 完成，输出文件：{OUTPUT_CSV_PATH}")
    print(f"COLLECT_TENSOR_INFO = {COLLECT_TENSOR_INFO}")
    print(f"COLLECT_TENSOR_ORIGIN_INFO = {COLLECT_TENSOR_ORIGIN_INFO}")


if __name__ == "__main__":
    main()
