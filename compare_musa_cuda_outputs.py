import argparse
import sys

import numpy as np


def load_named_arrays(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    names = data["names"].tolist()
    arrays = [data[f"arr_{i}"] for i in range(len(names))]
    return names, arrays


def numeric_metrics(a, b):
    finite_mask = np.isfinite(a) & np.isfinite(b)
    if np.any(finite_mask):
        a_f = a[finite_mask]
        b_f = b[finite_mask]
        diff = np.abs(a_f - b_f)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        denom = np.maximum(np.abs(b_f), 1e-12)
        max_rel = float(np.max(diff / denom))
    else:
        max_abs = 0.0
        mean_abs = 0.0
        max_rel = 0.0
    return max_abs, mean_abs, max_rel


def format_values(arr, max_elements):
    flat = arr.reshape(-1)
    show = flat[:max_elements]
    return np.array2string(show, threshold=max_elements, separator=", ")


def compare_outputs(
    musa_path,
    cuda_path,
    atol=1e-5,
    rtol=1e-4,
    topk=20,
    print_values=False,
    max_elements=16,
):
    musa_names, musa_arrays = load_named_arrays(musa_path)
    cuda_names, cuda_arrays = load_named_arrays(cuda_path)

    musa_map = {k: v for k, v in zip(musa_names, musa_arrays)}
    cuda_map = {k: v for k, v in zip(cuda_names, cuda_arrays)}

    common_names = sorted(set(musa_map.keys()) & set(cuda_map.keys()))
    only_musa = sorted(set(musa_map.keys()) - set(cuda_map.keys()))
    only_cuda = sorted(set(cuda_map.keys()) - set(musa_map.keys()))

    if only_musa:
        print("Only in MUSA outputs:")
        for name in only_musa:
            print(f"  {name}")
    if only_cuda:
        print("Only in CUDA outputs:")
        for name in only_cuda:
            print(f"  {name}")

    rows = []
    mismatch = 0
    for name in common_names:
        a = musa_map[name]
        b = cuda_map[name]
        if a.shape != b.shape:
            mismatch += 1
            rows.append((name, "shape_mismatch", np.inf, np.inf, np.inf, 0, 0, 0, 0))
            continue

        if np.issubdtype(a.dtype, np.number):
            musa_nan_count = int(np.isnan(a).sum())
            musa_inf_count = int(np.isinf(a).sum())
        else:
            musa_nan_count = 0
            musa_inf_count = 0

        if np.issubdtype(b.dtype, np.number):
            cuda_nan_count = int(np.isnan(b).sum())
            cuda_inf_count = int(np.isinf(b).sum())
        else:
            cuda_nan_count = 0
            cuda_inf_count = 0

        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            max_abs, mean_abs, max_rel = numeric_metrics(a, b)
            ok = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
            status = "ok" if ok else "not_close"
        else:
            # 非数值类型（如 string/bool）用严格相等比较
            ok = np.array_equal(a, b)
            status = "ok" if ok else "not_equal"
            max_abs = 0.0
            mean_abs = 0.0
            max_rel = 0.0

        if not ok:
            mismatch += 1
        rows.append(
            (
                name,
                status,
                max_abs,
                mean_abs,
                max_rel,
                musa_nan_count,
                musa_inf_count,
                cuda_nan_count,
                cuda_inf_count,
            )
        )

    rows_sorted = sorted(rows, key=lambda x: x[2], reverse=True)
    print(
        f"\nCompared tensors: {len(common_names)}, mismatch: {mismatch}, "
        f"atol={atol}, rtol={rtol}"
    )
    print("\nTop diff tensors:")
    for (
        name,
        status,
        max_abs,
        mean_abs,
        max_rel,
        musa_nan_count,
        musa_inf_count,
        cuda_nan_count,
        cuda_inf_count,
    ) in rows_sorted[:topk]:
        print(
            f"{status:12s} | max_abs={max_abs:.6e} | mean_abs={mean_abs:.6e} "
            f"| max_rel={max_rel:.6e} | "
            f"musa_nan={musa_nan_count} musa_inf={musa_inf_count} "
            f"cuda_nan={cuda_nan_count} cuda_inf={cuda_inf_count} | {name}"
        )
        if print_values:
            a = musa_map[name]
            b = cuda_map[name]
            print(f"  shape={a.shape}, dtype_musa={a.dtype}, dtype_cuda={b.dtype}")
            print(f"  musa[:{max_elements}]={format_values(a, max_elements)}")
            print(f"  cuda[:{max_elements}]={format_values(b, max_elements)}")

    return mismatch == 0 and (not only_musa) and (not only_cuda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musa-output", required=True, help="Path to MUSA output npz")
    parser.add_argument("--cuda-output", required=True, help="Path to CUDA output npz")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--print-values", action="store_true", help="Print tensor values for topk rows")
    parser.add_argument("--max-elements", type=int, default=16, help="Max flattened elements to print per tensor")
    args = parser.parse_args()

    ok = compare_outputs(
        musa_path=args.musa_output,
        cuda_path=args.cuda_output,
        atol=args.atol,
        rtol=args.rtol,
        topk=args.topk,
        print_values=args.print_values,
        max_elements=args.max_elements,
    )
    sys.exit(0 if ok else 1)

