import json
import csv
from collections import defaultdict

# 全局变量
INPUT_JSON_PATH = "/home/learlee/dev/projects/conv2d_silu_dev/test/build/yolov8s_om.json"
OUTPUT_CSV_PATH = "/home/learlee/dev/projects/conv2d_silu_dev/test/build/yolov8s_conv2d_info.csv"
# 开关：True收集x/filter/bias/y完整信息；False仅收集strides、pads、dilations，count只按这三个字段聚合
COLLECT_TENSOR_INFO = False


def get_shape_from_desc(desc):
    """从tensor desc提取shape dim列表，没有shape返回空列表"""
    if "shape" not in desc or "dim" not in desc["shape"]:
        return []
    return desc["shape"]["dim"]


def get_real_layout(desc):
    """取实际layout，不是origin_format"""
    return desc.get("layout", "")


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

            # x: input_desc index0
            x_desc = input_descs[0] if len(input_descs) > 0 else {}
            x_shape = get_shape_from_desc(x_desc)
            x_layout = get_real_layout(x_desc)

            # filter: input_desc index1
            filter_desc = input_descs[1] if len(input_descs) > 1 else {}
            filter_shape = get_shape_from_desc(filter_desc)
            filter_layout = get_real_layout(filter_desc)

            # bias: input_desc index2
            bias_desc = input_descs[2] if len(input_descs) > 2 else {}
            bias_shape = get_shape_from_desc(bias_desc)
            bias_layout = get_real_layout(bias_desc)

            # y: output_desc index0
            y_desc = output_descs[0] if len(output_descs) > 0 else {}
            y_shape = get_shape_from_desc(y_desc)
            y_layout = get_real_layout(y_desc)

            # 取conv属性 strides pads dilations
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

            # 根据开关构造聚合key
            if COLLECT_TENSOR_INFO:
                record_key = (
                    tuple(x_shape),
                    x_layout,
                    tuple(filter_shape),
                    filter_layout,
                    tuple(bias_shape),
                    bias_layout,
                    tuple(y_shape),
                    y_layout,
                    str(strides),
                    str(pads),
                    str(dilations),
                )
            else:
                # 关闭tensor信息，仅用strides/pads/dilations做聚合key
                record_key = (str(strides), str(pads), str(dilations))

            conv_records.append((record_key, x_shape, x_layout, filter_shape, filter_layout,
                                 bias_shape, bias_layout, y_shape, y_layout, strides, pads, dilations))

    # 统计count
    counter = defaultdict(int)
    raw_store = dict()  # key -> 原始各个字段值，用于输出
    for item in conv_records:
        rk, x_shp, x_ly, filt_shp, filt_ly, bias_shp, bias_ly, y_shp, y_ly, strides, pads, dilations = item
        counter[rk] += 1
        if rk not in raw_store:
            raw_store[rk] = (x_shp, x_ly, filt_shp, filt_ly, bias_shp, bias_ly, y_shp, y_ly, strides, pads, dilations)

    # csv表头根据开关动态变化
    if COLLECT_TENSOR_INFO:
        csv_header = [
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
    else:
        # 不收集tensor信息时，tensor相关列保留但填空字符串
        csv_header = [
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
        x_shp, x_ly, filt_shp, filt_ly, bias_shp, bias_ly, y_shp, y_ly, strides, pads, dilations = raw_store[key]
        if COLLECT_TENSOR_INFO:
            row = [
                list(x_shp),
                x_ly,
                list(filt_shp),
                filt_ly,
                list(bias_shp),
                bias_ly,
                list(y_shp),
                y_ly,
                str(strides),
                str(pads),
                str(dilations),
                cnt,
            ]
        else:
            # 关闭tensor收集，tensor字段全部填空，只保留strides/pads/dilations/count
            row = [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                str(strides),
                str(pads),
                str(dilations),
                cnt,
            ]
        rows.append(row)

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(csv_header)
        writer.writerows(rows)

    print(f"完成，输出文件：{OUTPUT_CSV_PATH}")
    if not COLLECT_TENSOR_INFO:
        print("当前模式：不收集x/filter/bias/y张量信息，仅按strides/pads/dilations聚合统计count")


if __name__ == "__main__":
    main()
