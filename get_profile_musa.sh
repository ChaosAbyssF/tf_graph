#!/bin/bash
set -e

# 默认值
GRAPH_PATH="meta_graph_1.spec"
BASE_OUTPUT_DIR="profile_musa"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --graph|-g)
            GRAPH_PATH="$2"
            shift 2
            ;;
        --output-dir|-o)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--graph PATH] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

# 提取模型名称（不含路径和 .pb 扩展名）
MODEL_NAME=$(basename "${GRAPH_PATH%.*}")

# 添加时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 构造最终输出目录
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

MSYS_OUTPUT="${OUTPUT_DIR}/msys_profile"

echo "========================= running msys ============================"
echo "Model:       ${GRAPH_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "---------------------------------------------------------------"

# 运行 profiling
msys profile \
    --device=0 \
    --trace=musa,osrt \
    --output="${MSYS_OUTPUT}" \
    python graph_runner_profile.py --graph "${GRAPH_PATH}" \
    --output "${OUTPUT_DIR}" \
    --platform "musa" \
    2>&1 | tee "${OUTPUT_DIR}/msys_profile.log"

# 生成 CSV 报告
msys stats \
    --format csv \
    --report musa_api_sum,musa_gpu_kern_sum,musa_kern_exec_sum,musa_api_gpu_sum \
    --output "${MSYS_OUTPUT}" \
    "${MSYS_OUTPUT}.msys-rep"

echo "==============================================================="
echo "✅ msys CSV reports exported to: ${OUTPUT_DIR}/"
echo "   Files: $(ls ${OUTPUT_DIR}/*.csv 2>/dev/null || echo 'none')"
