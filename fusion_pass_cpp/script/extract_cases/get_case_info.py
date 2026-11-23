import onnx
from onnx import shape_inference
from onnx.utils import extract_model
import json
import os
import subprocess
import csv
from collections import defaultdict
import tempfile


def _get_attr_value(attr_list, key):
    """从OM结构的attr列表中按键提取值"""
    for attr in attr_list:
        if attr["key"] == key:
            val = attr["value"]
            if "i" in val:
                return val["i"]
            elif "s" in val:
                return val["s"]
            elif "b" in val:
                return val["b"]
            elif "list" in val and "i" in val["list"]:
                return val["list"]["i"]
            elif "list" in val and "s" in val["list"]:
                return val["list"]["s"]
            else:
                return val
    return None


def _shape_to_str(shape):
    """将shape列表转为 [a, b, c] 格式字符串，匹配你要求的CSV记录样式"""
    if not shape:
        return ""
    return f"[{', '.join(map(str, shape))}]"


def extract_and_classify_patterns(original_onnx_path, output_dir):
    """
    第一步：从原始ONNX中匹配Conv+Sigmoid+Mul pattern，
    按参数分类并截取每类的代表子图，同时统计每类实例数count
    """
    print("===== 第一步：匹配并截取Pattern子图 =====")
    model = onnx.load(original_onnx_path)
    model = shape_inference.infer_shapes(model)

    # 临时保存带形状信息的模型，用于子图截取
    tmp_model_path = tempfile.mktemp(suffix=".onnx")
    onnx.save(model, tmp_model_path)

    # 构建辅助映射表
    value_info_map = {}
    for vi in model.graph.input:
        value_info_map[vi.name] = vi
    for vi in model.graph.output:
        value_info_map[vi.name] = vi
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi

    initializer_map = {init.name: init for init in model.graph.initializer}

    tensor_consumers = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            tensor_consumers[input_name].append(node)

    # 匹配目标pattern结构
    pattern_instances = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        conv_out = node.output[0]
        consumers = tensor_consumers.get(conv_out, [])
        if len(consumers) != 2:
            continue

        sigmoid_node = None
        mul_node = None
        for c in consumers:
            if c.op_type == "Sigmoid":
                sigmoid_node = c
            elif c.op_type == "Mul":
                mul_node = c

        if not sigmoid_node or not mul_node:
            continue

        sigmoid_out = sigmoid_node.output[0]
        if set(mul_node.input) == {conv_out, sigmoid_out}:
            pattern_instances.append((node, sigmoid_node, mul_node))

    print(f"共匹配到 {len(pattern_instances)} 个pattern实例")

    # 按参数分类，统计每类实例数count
    pattern_classes = {}
    for conv_node, sigmoid_node, mul_node in pattern_instances:
        # 提取分类维度：输入形状、权重形状、卷积核心参数
        input_feat_name = conv_node.input[0]
        if input_feat_name in value_info_map:
            input_shape = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in value_info_map[input_feat_name].type.tensor_type.shape.dim
            ]
        else:
            input_shape = None

        weight_name = conv_node.input[1]
        if weight_name in initializer_map:
            weight_shape = list(initializer_map[weight_name].dims)
        elif weight_name in value_info_map:
            weight_shape = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in value_info_map[weight_name].type.tensor_type.shape.dim
            ]
        else:
            weight_shape = None

        conv_attrs = {}
        for attr in conv_node.attribute:
            if attr.name == "dilations":
                conv_attrs["dilations"] = list(attr.ints)
            elif attr.name == "pads":
                conv_attrs["pads"] = list(attr.ints)
            elif attr.name == "strides":
                conv_attrs["strides"] = list(attr.ints)
            elif attr.name == "group":
                conv_attrs["group"] = attr.i

        class_key = (
            tuple(input_shape) if input_shape else None,
            tuple(weight_shape) if weight_shape else None,
            tuple(conv_attrs.get("dilations", [1, 1])),
            tuple(conv_attrs.get("pads", [0, 0, 0, 0])),
            tuple(conv_attrs.get("strides", [1, 1])),
            conv_attrs.get("group", 1)
        )

        if class_key not in pattern_classes:
            pattern_classes[class_key] = [conv_node, sigmoid_node, mul_node, 1]
        else:
            pattern_classes[class_key][3] += 1

    print(f"共分为 {len(pattern_classes)} 类不同的pattern")

    # 截取每类代表子图并保存
    os.makedirs(output_dir, exist_ok=True)
    classes_info = []
    for class_id, (class_key, (conv_node, sigmoid_node, mul_node, count)) in enumerate(pattern_classes.items()):
        model_name = f"pattern_class_{class_id}"
        onnx_save_path = os.path.join(output_dir, f"{model_name}.onnx")

        sub_inputs = [conv_node.input[0]]
        sub_outputs = [mul_node.output[0]]

        try:
            extract_model(tmp_model_path, onnx_save_path, sub_inputs, sub_outputs)
            classes_info.append({
                "class_id": class_id,
                "model_name": model_name,
                "count": count,
                "onnx_path": onnx_save_path
            })
            print(f"  类{class_id}: 实例数{count}，子图已保存")
        except Exception as e:
            print(f"  类{class_id} 截取子图失败: {str(e)}")

    if os.path.exists(tmp_model_path):
        os.remove(tmp_model_path)

    return classes_info


def compile_onnx_to_om(onnx_path, om_prefix, soc_version, atc_cmd="atc"):
    """第二步-1：将ONNX子图编译为OM模型"""
    cmd = [
        atc_cmd,
        "--framework=5",
        f"--model={onnx_path}",
        f"--soc_version={soc_version}",
        f"--output={om_prefix}",
        "--log=error"
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=300
        )
        if result.returncode == 0:
            om_path = f"{om_prefix}.om"
            if os.path.exists(om_path):
                return True, om_path, ""
            return False, "", "编译成功但未生成OM文件"
        return False, "", result.stderr.strip() or "编译失败"
    except subprocess.TimeoutExpired:
        return False, "", "编译超时"
    except FileNotFoundError:
        return False, "", "未找到atc命令"
    except Exception as e:
        return False, "", str(e)


def export_om_to_json(om_path, json_path, atc_cmd="atc"):
    """第二步-2：将OM模型导出为结构JSON文件"""
    cmd = [
        atc_cmd,
        "--mode=1",
        f"--om={om_path}",
        f"--json={json_path}",
        "--log=error"
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120
        )
        if result.returncode == 0:
            if os.path.exists(json_path):
                return True, ""
            return False, "导出成功但未生成JSON文件"
        return False, result.stderr.strip() or "导出失败"
    except subprocess.TimeoutExpired:
        return False, "导出超时"
    except Exception as e:
        return False, str(e)


def parse_om_struct_json(om_json_path, count):
    """
    第三步：解析OM结构JSON，严格提取指定字段
    区分原始逻辑形状（origin）和编译后硬件实际形状
    """
    try:
        with open(om_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph_list = data.get("graph", [])
        if not graph_list:
            return None
        graph = graph_list[0]
        ops = graph.get("op", [])

        # 定位Conv2D算子
        conv_op = None
        for op in ops:
            if op.get("type") == "Conv2D":
                conv_op = op
                break
        if not conv_op:
            return None

        input_desc = conv_op.get("input_desc", [])
        output_desc = conv_op.get("output_desc", [])
        if len(input_desc) < 3 or len(output_desc) < 1:
            return None

        # 1. 特征图输入 x
        x_desc = input_desc[0]
        x_attr = x_desc.get("attr", [])
        x_origin_shape = _get_attr_value(x_attr, "origin_shape")
        x_origin_layout = _get_attr_value(x_attr, "origin_format")
        x_shape = x_desc.get("shape", {}).get("dim", [])
        x_layout = x_desc.get("layout", "")

        # 2. 卷积核权重 filter
        filter_desc = input_desc[1]
        filter_attr = filter_desc.get("attr", [])
        filter_origin_shape = _get_attr_value(filter_attr, "origin_shape")
        filter_origin_layout = _get_attr_value(filter_attr, "origin_format")
        filter_shape = filter_desc.get("shape", {}).get("dim", [])
        filter_layout = filter_desc.get("layout", "")

        # 3. 偏置 bias
        bias_desc = input_desc[2]
        bias_attr = bias_desc.get("attr", [])
        bias_origin_shape = _get_attr_value(bias_attr, "origin_shape")
        bias_origin_layout = _get_attr_value(bias_attr, "origin_format")
        bias_shape = bias_desc.get("shape", {}).get("dim", [])
        bias_layout = bias_desc.get("layout", "")

        # 4. 卷积输出 y
        y_desc = output_desc[0]
        y_attr = y_desc.get("attr", [])
        y_origin_shape = _get_attr_value(y_attr, "origin_shape")
        y_origin_layout = _get_attr_value(y_attr, "origin_format")
        y_shape = y_desc.get("shape", {}).get("dim", [])
        y_layout = y_desc.get("layout", "")

        # 5. 卷积核心参数
        conv_attrs = conv_op.get("attr", [])
        strides = _get_attr_value(conv_attrs, "strides")
        pads = _get_attr_value(conv_attrs, "pads")
        dilations = _get_attr_value(conv_attrs, "dilations")

        # 组装结果，所有shape统一为 [a, b, c] 格式
        return {
            "x_origin_shape": _shape_to_str(x_origin_shape),
            "x_origin_layout": x_origin_layout if x_origin_layout else "",
            "filter_origin_shape": _shape_to_str(filter_origin_shape),
            "filter_origin_layout": filter_origin_layout if filter_origin_layout else "",
            "bias_origin_shape": _shape_to_str(bias_origin_shape),
            "bias_origin_layout": bias_origin_layout if bias_origin_layout else "",
            "y_origin_shape": _shape_to_str(y_origin_shape),
            "y_origin_layout": y_origin_layout if y_origin_layout else "",
            "x_shape": _shape_to_str(x_shape),
            "x_layout": x_layout if x_layout else "",
            "filter_shape": _shape_to_str(filter_shape),
            "filter_layout": filter_layout if filter_layout else "",
            "bias_shape": _shape_to_str(bias_shape),
            "bias_layout": bias_layout if bias_layout else "",
            "y_shape": _shape_to_str(y_shape),
            "y_layout": y_layout if y_layout else "",
            "strides": _shape_to_str(strides),
            "pads": _shape_to_str(pads),
            "dilations": _shape_to_str(dilations),
            "count": count
        }

    except Exception as e:
        print(f"    解析失败: {str(e)}")
        return None


def main():
    # ===================== 配置项 =====================
    # 原始YOLOv8s ONNX模型路径
    ORIGINAL_ONNX = "/home/learlee/dev/projects/conv2d_silu_dev/test/yolov8s.onnx"
    # 所有输出文件统一存放目录
    OUTPUT_DIR = "/home/learlee/dev/projects/conv2d_silu_dev/test/build/output_all"
    # 昇腾芯片型号
    SOC_VERSION = "Ascend910B3"
    # atc 命令路径
    ATC_CMD = "atc"
    # 是否保留OM结构JSON中间文件
    KEEP_OM_STRUCT_JSON = True
    # 调试开关：只处理前N个子图
    DEBUG_LIMIT = None  # 设为1则只跑1个，正式运行设为None
    # =================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 第一步：截取分类子图
    classes_info = extract_and_classify_patterns(ORIGINAL_ONNX, OUTPUT_DIR)
    if not classes_info:
        print("未截取到任何有效子图，程序退出")
        return

    # 调试限制数量
    if DEBUG_LIMIT is not None:
        classes_info = classes_info[:DEBUG_LIMIT]
        print(f"\n调试模式：仅处理前 {DEBUG_LIMIT} 个子图")

    # 第二步：批量编译OM + 导出结构JSON
    print("\n===== 第二步：编译OM并导出结构JSON =====")
    for cls_info in classes_info:
        model_name = cls_info["model_name"]
        onnx_path = cls_info["onnx_path"]
        om_prefix = os.path.join(OUTPUT_DIR, model_name)
        om_struct_json = os.path.join(OUTPUT_DIR, f"{model_name}_om_struct.json")

        print(f"处理 {model_name} ...")
        ok, om_path, err = compile_onnx_to_om(onnx_path, om_prefix, SOC_VERSION, ATC_CMD)
        if not ok:
            print(f"  ❌ OM编译失败: {err}")
            cls_info["compile_ok"] = False
            continue

        ok, err = export_om_to_json(om_path, om_struct_json, ATC_CMD)
        if not ok:
            print(f"  ❌ OM导出JSON失败: {err}")
            cls_info["compile_ok"] = False
            continue

        cls_info["om_path"] = om_path
        cls_info["om_struct_json"] = om_struct_json
        cls_info["compile_ok"] = True
        print(f"  ✅ 编译并导出成功")

        # 可选：删除中间结构JSON
        if not KEEP_OM_STRUCT_JSON and os.path.exists(om_struct_json):
            os.remove(om_struct_json)

    # 第三步：解析所有OM信息，汇总到CSV
    print("\n===== 第三步：汇总信息到CSV =====")
    csv_path = os.path.join(OUTPUT_DIR, "pattern_summary.csv")
    # 严格遵循你指定的字段顺序
    csv_fields = [
        "model_name", "om_file",
        "x_origin_shape", "x_origin_layout",
        "filter_origin_shape", "filter_origin_layout",
        "bias_origin_shape", "bias_origin_layout",
        "y_origin_shape", "y_origin_layout",
        "x_shape", "x_layout",
        "filter_shape", "filter_layout",
        "bias_shape", "bias_layout",
        "y_shape", "y_layout",
        "strides", "pads", "dilations",
        "count"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for cls_info in classes_info:
            if not cls_info.get("compile_ok", False):
                continue

            model_name = cls_info["model_name"]
            om_file = os.path.basename(cls_info["om_path"])
            count = cls_info["count"]

            fields = parse_om_struct_json(cls_info["om_struct_json"], count)
            if not fields:
                print(f"  {model_name} 字段解析失败，跳过")
                continue

            row = {"model_name": model_name, "om_file": om_file}
            row.update(fields)
            writer.writerow(row)
            print(f"  {model_name} 已写入CSV")

    print(f"\n全部处理完成")
    print(f"所有文件目录: {OUTPUT_DIR}")
    print(f"汇总CSV路径: {csv_path}")


if __name__ == "__main__":
    main()
