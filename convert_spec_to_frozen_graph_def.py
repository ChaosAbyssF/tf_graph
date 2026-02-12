import tensorflow as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

tf.compat.v1.disable_eager_execution()


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


if __name__ == "__main__":
    # input spec path
    spec_path = "meta_graph_1.spec"

    # build graph from spec
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.train.import_meta_graph(spec_path)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        output_op_names = []
        output_op_types = set()
        input_nodes = []
        input_nodes_dtype = []
        for op in graph.get_operations():
            if op.type == "Placeholder":
                input_nodes.append(op.name)
                input_nodes_dtype.append(op.outputs[0].dtype.as_datatype_enum)
            if op.type not in {"NoOp", "Assert", "Print"} and all(
                len(out.consumers()) == 0 for out in op.outputs
            ):
                root_op_set = get_root_upstream_op_types(op)
                if root_op_set - {"Const", "VariableV2"} == set():
                    # 全是 Const / VariableV2 上游的 op，跳过
                    continue
                output_op_names.append(op.name)
                output_op_types.add(op.type)

        output_op_names = list(set(output_op_names))
        print("AUTO outputs:", output_op_names)
        print("AUTO output op types:", output_op_types)
        # build frozen graph from output ops
        frozen_graph = convert_variables_to_constants(
            sess, sess.graph_def, output_op_names
        )

    # optimize frozen graph
    optimized_graph_def = optimize_for_inference(
        frozen_graph,
        input_node_names=input_nodes,
        output_node_names=output_op_names,
        placeholder_type_enum=input_nodes_dtype,
    )
    # save frozen graph
    with open(spec_path.replace(".spec", "_frozen.pb"), "wb") as f:
        f.write(optimized_graph_def.SerializeToString())
    print("Frozen graph saved to", spec_path.replace(".spec", "_frozen.pb"))
    # get op set and save to txt
    node_set = set()
    for node in optimized_graph_def.node:
        node_set.add(node.op)
    list_node = list(node_set)
    list_node.sort()
    with open(spec_path.replace(".spec", "_frozen_opset.txt"), "w") as f:
        for node in list_node:
            f.write(node + "\n")
    print("Op set saved to", spec_path.replace(".spec", "_frozen_opset.txt"))
