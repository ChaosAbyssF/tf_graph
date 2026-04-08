import argparse
import csv
import os

from tensorflow.core.framework import graph_pb2, types_pb2


DEFAULT_PBS = [
    "meta_graph_1_frozen.pb",
    "meta_graph_2_frozen.pb",
    "meta_graph_3_frozen.pb",
]


def load_graph_def(pb_path):
    graph_def = graph_pb2.GraphDef()
    with open(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def dtype_name(dtype_id):
    try:
        return types_pb2.DataType.Name(dtype_id)
    except Exception:
        return str(dtype_id)


def extract_node_dtype(node):
    dtype_keys = ["dtype", "T", "DstT", "SrcT", "Tout", "output_type", "out_type"]
    dtypes = []

    for key in dtype_keys:
        if key not in node.attr:
            continue
        attr_val = node.attr[key]
        if attr_val.HasField("type"):
            dtypes.append(dtype_name(attr_val.type))
        elif attr_val.HasField("list") and attr_val.list.type:
            dtypes.extend(dtype_name(t) for t in attr_val.list.type)

    if not dtypes:
        return ""
    return "|".join(sorted(set(dtypes)))


def shape_to_string(shape_proto):
    dims = []
    for dim in shape_proto.dim:
        if dim.size >= 0:
            dims.append(str(dim.size))
        else:
            dims.append("?")
    return "[" + ",".join(dims) + "]"


def extract_output_shapes(node):
    if "_output_shapes" not in node.attr:
        return ""
    attr_val = node.attr["_output_shapes"]
    if not attr_val.HasField("list") or not attr_val.list.shape:
        return ""
    return "|".join(shape_to_string(shape) for shape in attr_val.list.shape)


def dump_single_pb(pb_path, output_csv):
    graph_def = load_graph_def(pb_path)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pb_file",
                "node_name",
                "op_type",
                "dtype",
                "output_shapes",
                "device",
                "input_count",
                "inputs",
                "attr_keys",
            ]
        )

        for node in graph_def.node:
            writer.writerow(
                [
                    os.path.basename(pb_path),
                    node.name,
                    node.op,
                    extract_node_dtype(node),
                    extract_output_shapes(node),
                    node.device,
                    len(node.input),
                    "|".join(node.input),
                    "|".join(sorted(node.attr.keys())),
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump all node info (name/op/dtype/etc.) for frozen pb files."
    )
    parser.add_argument(
        "--pbs",
        type=str,
        default=",".join(DEFAULT_PBS),
        help="Comma-separated pb list. Default: meta_graph_1/2/3_frozen.pb",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="node_info",
        help="Output directory for csv files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pb_paths = [item.strip() for item in args.pbs.split(",") if item.strip()]
    if not pb_paths:
        raise ValueError("No pb file provided in --pbs.")

    for pb_path in pb_paths:
        if not os.path.exists(pb_path):
            raise FileNotFoundError(pb_path)
        csv_name = os.path.splitext(os.path.basename(pb_path))[0] + "_nodes.csv"
        output_csv = os.path.join(args.output_dir, csv_name)
        dump_single_pb(pb_path, output_csv)
        print(f"Saved node info: {output_csv}")


if __name__ == "__main__":
    main()
