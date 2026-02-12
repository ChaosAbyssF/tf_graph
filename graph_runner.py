import tensorflow as tf
import numpy as np
import os
from tensorflow.core.framework import graph_pb2


tf.compat.v1.disable_eager_execution()

musa_plugin_path = "/xxxx/tensorflow_musa_extension/build/libmusa_plugin.so"

# ==========================================
# 1. 加载 MUSA 插件
# ==========================================
def load_musa_plugin():
    if os.path.exists(musa_plugin_path):
        try:
            tf.load_op_library(musa_plugin_path)
            print(f">>>> [MUSA] Plugin loaded successfully from: {musa_plugin_path}")
        except Exception as e:
            print(f"!!!! [MUSA] Failed to load plugin: {e}")
    else:
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}, assuming built-in.")

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
    if dtype == tf.float32:
        data = np.random.rand(*shape)
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        return data
    elif dtype == tf.float16:
        return np.random.rand(*shape).astype(np.float16)
    elif dtype == tf.int32:
        return np.random.randint(0, 100, size=shape, dtype=np.int32)
    elif dtype == tf.int64:
        return np.random.randint(0, 100, size=shape, dtype=np.int64)
    elif dtype == tf.bool:
        return np.random.choice([True, False], size=shape)
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


def run_inference(pb_path):
    load_musa_plugin()
    graph_def = load_graph_def(pb_path)
    graph = import_graph(graph_def)

    # 1. 创建配置
    config = tf.compat.v1.ConfigProto()

    # 允许自动软放置（关键：如果某个操作 GPU 不支持，自动转 CPU，防止报错）
    config.allow_soft_placement = True

    # 打印设备日志（可选：运行代码时可以看到操作到底被分配到了哪里，方便调试）
    config.log_device_placement = True

    # 显存按需分配（可选：防止 TF 一次性占满所有显存）
    config.gpu_options.allow_growth = True

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
        output_values = sess.run(outputs, feed_dict=feed_dict)

    print("\n===== Inference Done =====")
    for out, val in zip(outputs, output_values):
        print(f"{out.name} -> output shape: {val.shape}")

    return output_values


if __name__ == "__main__":
    spec_path = "meta_graph_1.spec"
    pb_path = "meta_graph_1_frozen.pb"
    batch_size = 1
    run_inference(pb_path)
