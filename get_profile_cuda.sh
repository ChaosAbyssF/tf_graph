#!/bin/bash

set -e

# =========================
# 0) 基础配置
# =========================

GRAPH_PATH="meta_graph_1.spec"
BASE_OUTPUT_DIR="profile_cuda"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

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
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODEL_NAME=$(basename "${GRAPH_PATH%.*}")

echo "=== Model: ${MODEL_NAME}"

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

NSYS_OUTPUT="${OUTPUT_DIR}/nsys_profile"
NCU_OUTPUT="${OUTPUT_DIR}/ncu_profile"

TEMP_LOG="${OUTPUT_DIR}/temp_profile.log"

echo "Output dir: ${OUTPUT_DIR}"

# =========================
# 1) 运行命令
# =========================

GRAPH_CMD="python ./graph_runner_profile.py \
  --graph ${GRAPH_PATH} \
  --output ${OUTPUT_DIR} \
  --platform cuda"

# =========================
# 2) Nsight Systems
# =========================

echo "=== [Step1] Nsight Systems profiling ==="

if nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output=${NSYS_OUTPUT} \
    ${GRAPH_CMD} > "$TEMP_LOG" 2>&1; then

    cat "$TEMP_LOG" | tee "${OUTPUT_DIR}/nsys_profile.log"

    REPORT_FILE="${NSYS_OUTPUT}.nsys-rep"

    if [ ! -f "$REPORT_FILE" ]; then
        echo "nsys report missing!"
        exit 1
    fi

    echo "=== [Step2] Export nsys CSV ==="

    nsys stats \
        --report cuda_api_sum,cuda_gpu_kern_sum \
        --format csv \
        --output ${NSYS_OUTPUT} \
        "$REPORT_FILE"

    echo "nsys export done."

else

    cat "$TEMP_LOG"
    echo "nsys profiling failed"
    exit 1

fi

rm -f "$TEMP_LOG"

# =========================
# 3) Nsight Compute
# =========================

# echo "=== [Step3] Nsight Compute profiling ==="

# NCU_LOG="${OUTPUT_DIR}/ncu_profile.log"

# if ncu \
#   --mode=launch-and-attach \
#   --profile-from-start on \
#   --target-processes all \
#   --nvtx \
#   --kernel-name-base demangled \
#   --set speedOfLight \
#   --launch-skip 10 \
#   --launch-count 50 \
#   --force-overwrite \
#   --export ${NCU_OUTPUT} \
#   ${GRAPH_CMD} > "$NCU_LOG" 2>&1; then

#     echo "ncu profile finished"

# else

#     echo "ncu profiling failed"
#     cat "$NCU_LOG"
#     exit 1

# fi

# # =========================
# # 4) 导出 CSV
# # =========================

# NCU_REPORT="${NCU_OUTPUT}.ncu-rep"

# if [ ! -f "$NCU_REPORT" ]; then
#     echo "ncu report not found"
#     exit 1
# fi

# echo "=== [Step4] Export ncu CSV ==="

# ncu \
#     --import "$NCU_REPORT" \
#     --csv \
#     --page details \
#     > "${NCU_OUTPUT}.csv"


# echo "================================================"
# echo "Profiling finished"

# echo "NSYS report:"
# echo "   ${NSYS_OUTPUT}.nsys-rep"

# echo "NCU report:"
# echo "   ${NCU_REPORT}"

# echo "CSV files:"
# ls ${OUTPUT_DIR}/*.csv 2>/dev/null