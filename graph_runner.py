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
    np_state = np.random.RandomState(2026)
    
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


def save_latency_to_csv(csv_path, latencies_ms, warmup_runs, num_runs):
    avg_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "run_idx", "latency_ms"])
        for idx, latency in enumerate(latencies_ms, start=1):
            writer.writerow(["run", idx, f"{latency:.6f}"])

        writer.writerow([])
        writer.writerow(["summary", "metric", "value"])
        writer.writerow(["summary", "warmup_runs", warmup_runs])
        writer.writerow(["summary", "num_runs", num_runs])
        writer.writerow(["summary", "avg_ms", f"{avg_ms:.6f}"])
        writer.writerow(["summary", "p50_ms", f"{p50_ms:.6f}"])
        writer.writerow(["summary", "p95_ms", f"{p95_ms:.6f}"])
        writer.writerow(["summary", "min_ms", f"{min_ms:.6f}"])
        writer.writerow(["summary", "max_ms", f"{max_ms:.6f}"])

    print(f"\n===== Latency CSV Saved =====\n{csv_path}")


def run_inference(pb_path, warmup_runs=10, num_runs=50, latency_csv_path=None, platform='cpu'):
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
    
     # ================= 调试打印部分 (修正版) =================
    print("\n===== Verifying Feed Dict Shapes =====")
    for tensor, data in feed_dict.items():
        # 1. 安全获取 Graph 中的形状
        try:
            # 尝试转换为列表
            graph_shape = tensor.shape.as_list()
        except ValueError:
            # 如果形状完全未知 (None)，捕获异常并标记
            graph_shape = "Unknown (None)"
        
        # 2. 安全获取实际数据的形状
        import numpy as np
        if not isinstance(data, np.ndarray):
            # 如果是标量 (float/int)，转为 numpy 数组以便统一处理
            data = np.array(data)
            feed_dict[tensor] = data  # 更新字典，防止 sess.run 报错
        
        actual_shape = list(data.shape)
        dtype_info = data.dtype

        print(f"Input: {tensor.name}")
        print(f"  -> Graph Shape: {graph_shape}")
        print(f"  -> Actual Data Shape: {actual_shape} (dtype: {dtype_info})")
        
        # 只有当 Graph 形状已知时，才进行维度检查
        if graph_shape != "Unknown (None)":
            if None not in graph_shape:
                if len(graph_shape) != len(actual_shape):
                    print(f"  [WARNING] Dimension count mismatch! Graph expects {len(graph_shape)}D, got {len(actual_shape)}D")
            else:
                # 形状中包含 None (动态维)，只打印提示
                print(f"  [INFO] Graph shape contains dynamic dimensions (None). Matching based on known dims.")


    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        print("\n===== Warmup Run =====")
        for _ in range(warmup_runs):
            sess.run(outputs, feed_dict=feed_dict)

        latencies_ms = []
        output_values = None
        print("\n===== Inference Run =====")
        for _ in range(num_runs):
            start = time.perf_counter()
            output_values = sess.run(outputs, feed_dict=feed_dict)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    avg_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))

    print("\n===== Inference Done =====")
    for out, val in zip(outputs, output_values):
        print(f"{out.name} -> output shape: {val.shape} | sample={val.flat[:6] if val.size > 0 else []}")
    print("\n===== Latency Stats (ms) =====")
    print(f"warmup_runs={warmup_runs}, num_runs={num_runs}")
    print(
        f"avg={avg_ms:.3f}, p50={p50_ms:.3f}, p95={p95_ms:.3f}, "
        f"min={min_ms:.3f}, max={max_ms:.3f}"
    )

    if latency_csv_path:
        save_latency_to_csv(latency_csv_path, latencies_ms, warmup_runs, num_runs)

    return output_values, latencies_ms


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen PB inference and benchmark latency.")
    parser.add_argument("--graph", "--g", default="meta_graph_1.spec", help="Path to frozen pb file.")
    parser.add_argument("--batch-size", "--bs", type=int, default=1024, help="Batch size for dynamic first dimension.")
    parser.add_argument('--platform', type=str, choices=['cpu', 'cuda', 'musa'],
                        default='cpu', help='Target platform for inference.')
    parser.add_argument("--warmup-runs", type=int, default=10, help="Warmup iteration count.")
    parser.add_argument("--num-runs", type=int, default=50, help="Measured iteration count.")
    parser.add_argument(
        "--latency-csv",
        default="latency_stats.csv",
        help="Output CSV path for per-run latency and summary stats. Set empty string to disable.",
    )
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
    pb_path = os.path.splitext(args.graph)[0] + "_frozen.pb"
    
    batch_size = args.batch_size
    csv_name, ext = os.path.splitext(args.latency_csv)
    
    model_name = os.path.splitext(spec_path)[0]
    output_path = args.output if args.output else model_name
    
    if os.path.exists(output_path):
        print("Using existing output path:", output_path)
    else:
        print("Creating output path:", output_path)
        os.makedirs(output_path)
        
    latency_csv_path = os.path.join(output_path, "{}_batch_{}{}".format(csv_name, batch_size, ext))
    if os.path.exists(spec_path) and os.path.exists(pb_path):
        print("Using existing frozen graph:", pb_path)
        run_inference(
        pb_path=pb_path,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        latency_csv_path=latency_csv_path,
        platform=args.platform,
        )
    else:
        print("Converting spec to frozen graph by: python convert_spec_to_frozen_graph_def.py")
        


