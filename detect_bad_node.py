import tensorflow as tf
import numpy as np
import os
import time
import csv
import argparse
from tensorflow.core.framework import graph_pb2


tf.compat.v1.disable_eager_execution()

musa_plugin_path = "../tensorflow_musa_extension/build/libmusa_plugin.so"

# ==========================================
# 1. 加载 MUSA 插件
# ==========================================
def load_musa_plugin():
    if musa_plugin_path and os.path.exists(musa_plugin_path):
        try:
            tf.load_op_library(musa_plugin_path)
            print(f">>>> [MUSA] Plugin loaded successfully from: {musa_plugin_path}")
        except Exception as e:
            print(f"!!!! [MUSA] Failed to load plugin: {e}")
    else:
        print("[Info] MUSA Plugin loading skipped. Running on CPU.")
        
def load_graph_def(pb_path):
    graph_def = graph_pb2.GraphDef()
    with open(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        if node.device:
            node.device = ""
    return graph_def


def import_graph(graph_def):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph

def scan_placeholders(graph):
    global batch_size
    input_names = []
    for op in graph.get_operations():
        if op.type == "Placeholder":
            input_names.append(op.outputs[0].name)
    placeholders = []
    meta_graph = tf.Graph()
    with tf.compat.v1.Session(graph=meta_graph) as sess:
        tf.compat.v1.train.import_meta_graph(
            spec_path,
            clear_devices=False,
        )
        input_tensors = meta_graph.get_collection("input_spec")
        for input_tensor in input_tensors:
            if input_tensor.name in input_names:
                shape = input_tensor.shape.as_list()
                if len(shape) > 0 and shape[0] is None:
                    shape[0] = batch_size
                placeholders.append(
                    {
                        "name": input_tensor.name,
                        "dtype": input_tensor.dtype,
                        "shape": shape,
                    }
                )
    return placeholders


def generate_random_input(name, dtype, shape):
    """根据 dtype + shape 自动造数据"""
    
    # 固定随机数种子
    np_state = np.random.RandomState(42)
    
    if dtype == tf.float32:
        data = np_state.rand(*shape)
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        return data
    elif dtype == tf.float16:
        return np_state.rand(*shape).astype(np.float16)
    elif dtype == tf.int32:
        return np_state.randint(0, 100, size=shape, dtype=np.int32)
    elif dtype == tf.int64:
        return np_state.randint(0, 100, size=shape, dtype=np.int64)
    elif dtype == tf.bool:
        return np_state.choice([True, False], size=shape)
    elif dtype == tf.string:
        return np.array([b"aweme_dou_plus" for _ in range(np.prod(shape))]).reshape(
            shape
        )
    else:
        raise ValueError(f"Unsupported dtype {dtype} for placeholder {name}")


def build_feed_dict(graph, placeholders):
    feed_dict = {}
    for ph in placeholders:
        tensor = graph.get_tensor_by_name(ph["name"])
        data = generate_random_input(ph["name"], ph["dtype"], ph["shape"])
        feed_dict[tensor] = data
    return feed_dict


def get_root_upstream_op_types(op):
    """
    输入:
        op: tf.Operation
    输出:
        set[str]，所有最上游的 op.type
    """
    visited = set()
    roots = set()

    def dfs(cur_op):
        if not cur_op.inputs:
            roots.add(cur_op.type)
            return

        for tensor in cur_op.inputs:
            src_op = tensor.op
            if src_op not in visited:
                visited.add(src_op)
                dfs(src_op)

    dfs(op)
    return roots


def find_output_tensors(graph):
    """自动找输出节点：没有 consumer 的 tensor"""
    global batch_size
    outputs = []
    for op in graph.get_operations():
        if op.type not in {"NoOp", "Assert", "Print"} and all(
            len(out.consumers()) == 0 for out in op.outputs
        ):
            root_op_set = get_root_upstream_op_types(op)
            if root_op_set - {"Const", "VariableV2"} == set():
                # 全是 Const / VariableV2 上游的 op，跳过
                continue
            for out in op.outputs:
                outputs.append(out)
    return outputs

def contains_nan_inf(x):

    if not isinstance(x, np.ndarray):
        return False

    if not np.issubdtype(x.dtype, np.floating):
        return False

    return np.any(np.isnan(x)) or np.any(np.isinf(x))
def detect_first_nan_inf_node(sess, graph, feed_dict, output_dir):
    import numpy as np
    import json

    print("\n===== Scanning graph for NaN / Inf =====")

    for op in graph.get_operations():

        if not op.outputs:
            continue

        try:
            outputs = sess.run(op.outputs, feed_dict=feed_dict)
        except Exception:
            continue

        for idx, out in enumerate(outputs):

            if not isinstance(out, np.ndarray):
                continue

            if contains_nan_inf(out):

                print("\n!!!! Found NaN/Inf in node !!!!")
                print("Node name:", op.name)
                print("Op type:", op.type)

                input_info = []

                all_inputs_valid = True

                for tensor in op.inputs:

                    try:
                        val = sess.run(tensor, feed_dict=feed_dict)
                    except Exception:
                        val = None

                    info = {
                            "name": tensor.name,
                            "dtype": str(tensor.dtype),
                            "shape": str(tensor.shape),
                            "source_op_name": tensor.op.name,
                            "source_op_type": tensor.op.type,
                            "sample": None
                        }

                    if isinstance(val, np.ndarray):
                        info["sample"] = val.flat[:6].tolist()

                        if contains_nan_inf(val):
                            all_inputs_valid = False

                    input_info.append(info)

                if not all_inputs_valid:
                    print("Input already contains NaN/Inf, skipping...")
                    continue

                output_info = {
                        "name": op.outputs[idx].name,
                        "dtype": str(op.outputs[idx].dtype),
                        "shape": str(op.outputs[idx].shape),
                        "op_name": op.name,
                        "op_type": op.type,
                        "sample": out.flat[:6].tolist()
                    }

                record = {
                    "node_name": op.name,
                    "op_type": op.type,
                    "inputs": input_info,
                    "output": output_info
                }

                os.makedirs(output_dir, exist_ok=True)

                save_path = os.path.join(output_dir, "first_nan_inf_node.json")

                with open(save_path, "w") as f:
                    json.dump(record, f, indent=2)

                print("Saved debug info to:", save_path)

                return record

    print("No NaN/Inf node found.")
    return None

def run_inference(pb_path, platform='cpu'):
    graph_def = load_graph_def(pb_path)
    graph = import_graph(graph_def)

    # 1. 创建配置
    config = tf.compat.v1.ConfigProto()

    # 允许自动软放置（关键：如果某个操作 GPU 不支持，自动转 CPU，防止报错）
    config.allow_soft_placement = True

    # 打印设备日志（可选：运行代码时可以看到操作到底被分配到了哪里，方便调试）
    config.log_device_placement = True

    if platform.lower() == 'cpu':
        # 正确设置CPU模式，不使用GPU
        config.device_count['GPU'] = 0
        print("Configured for CPU execution.")
    elif platform.lower() in ['cuda', 'musa']:
        # 显存按需分配（可选：防止 TF 一次性占满所有显存）
        config.gpu_options.allow_growth = True
        print(f"Configured for {platform.upper()} execution. Available devices will be used automatically.")
        if platform.lower() == 'musa':
            # MUSA 插件加载后，框架应能识别 MUSA 设备
            load_musa_plugin()
        else:
            cuda_path = "/usr/local/cuda" 
            if os.path.exists(cuda_path):
                os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
            else:
                print("CUDA path not found. Please check your CUDA installation.")
                exit(1)
            print("TF Version:", tf.__version__)
            print("GPU Devices:", tf.config.list_physical_devices('GPU'))

    placeholders = scan_placeholders(graph)
    outputs = find_output_tensors(graph)

    print("\n===== Placeholders =====")
    for ph in placeholders:
        print(f"{ph['name']} | dtype={ph['dtype']} | shape={ph['shape']}")

    print("\n===== Output Tensors =====")
    for out in outputs:
        print(f"{out.name} | dtype={out.dtype} | shape={out.shape}")

    feed_dict = build_feed_dict(graph, placeholders)

    with tf.compat.v1.Session(graph=graph, config=config) as sess:

        print("\n===== Detecting NaN / Inf nodes =====")

        detect_first_nan_inf_node(
            sess,
            graph,
            feed_dict,
            output_dir=output_path
        )

        print("\n===== Inference Run =====")
        output_values = sess.run(outputs, feed_dict=feed_dict)

        print("\n===== Inference Done =====")
    
    for out, val in zip(outputs, output_values):
        print(f"{out.name} -> output shape: {val.shape} | sample={val.flat[:6] if val.size > 0 else []}")

    return output_values


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen PB inference and benchmark latency.")
    parser.add_argument("--graph", "--g", default="meta_graph_1.spec", help="Path to frozen pb file.")
    parser.add_argument("--batch-size", "--bs", type=int, default=1, help="Batch size for dynamic first dimension.")
    parser.add_argument('--platform', type=str, choices=['cpu', 'cuda', 'musa'],
                        default='cpu', help='Target platform for inference.')
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path for per-run latency and summary stats. Set empty string to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # print args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    spec_path = args.graph
    pb_path = os.path.splitext(args.graph)[0] + "_frozen_fixed.pb"
    
    batch_size = args.batch_size
    model_name = os.path.splitext(spec_path)[0]
    output_path = args.output if args.output else model_name
    
    if os.path.exists(output_path):
        print("Using existing output path:", output_path)
    else:
        print("Creating output path:", output_path)
        os.makedirs(output_path)
        
    if os.path.exists(spec_path) and os.path.exists(pb_path):
        print("Using existing frozen graph:", pb_path)
        run_inference(
        pb_path=pb_path,
        platform=args.platform,
        )
    else:
        print("Converting spec to frozen graph by: python convert_spec_to_frozen_graph_def.py")
        



