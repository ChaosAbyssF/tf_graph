import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import graph_pb2


def load_graph_def(pb_path):
    graph_def = graph_pb2.GraphDef()
    with open(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def is_float_dtype(dtype):
    try:
        tf_dtype = tf.as_dtype(dtype)
    except TypeError:
        return False
    return tf_dtype.is_floating


def _tiny_for_dtype(dtype):
    try:
        return float(np.finfo(dtype).tiny)
    except Exception:
        # Fallback to float32 tiny if dtype info is unavailable (e.g., bfloat16 on some numpy builds)
        return float(np.finfo(np.float32).tiny)


def repair_array(arr):
    """
    Repair NaN/Inf without simply zeroing out.
    Strategy:
      - NaN -> mean of finite values
      - +Inf -> max of finite values
      - -Inf -> min of finite values
      - If no finite values exist: fill with tiny positive value
    """
    arr_fixed = np.array(arr, copy=True)
    finite_mask = np.isfinite(arr_fixed)
    if finite_mask.any():
        finite_vals = arr_fixed[finite_mask].astype(np.float64)
        mean_val = float(finite_vals.mean())
        max_val = float(finite_vals.max())
        min_val = float(finite_vals.min())
        arr_fixed[np.isnan(arr_fixed)] = mean_val
        arr_fixed[np.isposinf(arr_fixed)] = max_val
        arr_fixed[np.isneginf(arr_fixed)] = min_val
        strategy = {
            "nan_replacement": "mean_finite",
            "posinf_replacement": "max_finite",
            "neginf_replacement": "min_finite",
            "mean_finite": mean_val,
            "max_finite": max_val,
            "min_finite": min_val,
        }
    else:
        tiny = _tiny_for_dtype(arr_fixed.dtype)
        arr_fixed = np.full(arr_fixed.shape, tiny, dtype=arr_fixed.dtype)
        strategy = {
            "nan_replacement": "tiny",
            "posinf_replacement": "tiny",
            "neginf_replacement": "tiny",
            "tiny_value": tiny,
        }
    return arr_fixed, strategy


def scan_and_fix(graph_def, fix=False):
    findings = []
    fixed_count = 0

    for node in graph_def.node:
        if node.op != "Const":
            continue
        if "value" not in node.attr:
            continue
        tensor = node.attr["value"].tensor
        try:
            arr = tensor_util.MakeNdarray(tensor)
        except Exception:
            # Skip tensors that can't be materialized
            continue
        if not is_float_dtype(arr.dtype.type):
            continue

        nan_mask = np.isnan(arr)
        posinf_mask = np.isposinf(arr)
        neginf_mask = np.isneginf(arr)
        has_nan = nan_mask.any()
        has_inf = (posinf_mask | neginf_mask).any()
        if not (has_nan or has_inf):
            continue

        info = {
            "node_name": node.name,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
            "nan_count": int(nan_mask.sum()),
            "posinf_count": int(posinf_mask.sum()),
            "neginf_count": int(neginf_mask.sum()),
        }
        findings.append(info)

        if fix:
            arr_fixed, strategy = repair_array(arr)
            node.attr["value"].tensor.CopyFrom(
                tf.make_tensor_proto(arr_fixed, dtype=arr_fixed.dtype)
            )
            info["repair_strategy"] = strategy
            fixed_count += 1

    return findings, fixed_count


def main():
    parser = argparse.ArgumentParser(description="Scan and fix NaN/Inf in Const nodes of frozen PB")
    parser.add_argument("--pb", required=True, help="Path to frozen pb")
    parser.add_argument(
        "--out",
        default=None,
        help="Output fixed pb path (default: *_fixed.pb)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output report json path (default: *_nan_inf_report.json)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes by replacing NaN/Inf with 0 in Const nodes",
    )
    args = parser.parse_args()

    pb_path = args.pb
    if not os.path.exists(pb_path):
        raise FileNotFoundError(pb_path)

    base, ext = os.path.splitext(pb_path)
    out_path = args.out or f"{base}_fixed{ext}"
    report_path = args.report or f"{base}_nan_inf_report.json"

    graph_def = load_graph_def(pb_path)
    findings, fixed_count = scan_and_fix(graph_def, fix=args.fix)
    post_fix_findings = []
    if args.fix:
        post_fix_findings, _ = scan_and_fix(graph_def, fix=False)

    report = {
        "pb": pb_path,
        "fix_applied": bool(args.fix),
        "nan_inf_const_nodes": findings,
        "nan_inf_const_node_count": len(findings),
        "fixed_const_node_count": fixed_count,
        "post_fix_nan_inf_const_nodes": post_fix_findings,
        "post_fix_nan_inf_const_node_count": len(post_fix_findings),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.fix:
        with open(out_path, "wb") as f:
            f.write(graph_def.SerializeToString())

    print("Report saved to", report_path)
    if args.fix:
        print("Fixed frozen graph saved to", out_path)


if __name__ == "__main__":
    main()
