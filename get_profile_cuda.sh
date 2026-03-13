#!/bin/bash

# 0) 设置基础目录
GRAPH_PATH="meta_graph_1.spec"
BASE_OUTPUT_DIR="profile_cuda"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

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

 # 1) 提取模型名称（不含路径和 .pb 扩展名）
 MODEL_NAME=$(basename "${GRAPH_PATH%.*}")
 echo "=== 模型名称: ${MODEL_NAME} ==="

 # 构造最终输出目录
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

NSYS_OUTPUT="${OUTPUT_DIR}/nsys_profile"
NCU_OUTPUT="${OUTPUT_DIR}/ncu_profile"

TEMP_LOG="${OUTPUT_DIR}/temp_profile.log"

echo "=== [Step 0] 目录准备完成: ${OUTPUT_DIR} ==="

# 1) 定义要运行的图形/训练命令
# 请确保 python 路径和脚本路径正确
GRAPH_CMD="python ./graph_runner_profile.py --graph ${GRAPH_PATH} --output ${OUTPUT_DIR} --platform cuda"

echo "=== [Step 1] 开始 Nsight Systems (nsys) 采集 ==="
# 注意：nsys profile 成功后会生成 .nsys-rep 文件
# 如果 graph_runner.py 运行时间很长，nsys 会一直运行直到它结束
if nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output=${NSYS_OUTPUT} \
    ${GRAPH_CMD} > "$TEMP_LOG" 2>&1; then

    # 如果 nsys profile 成功，再将临时日志文件的内容复制到最终日志，并打印到屏幕上
    cat "$TEMP_LOG" | tee "${OUTPUT_DIR}/nsys_profile.log"
    echo "✅ nsys profile completed successfully and log was written."
    
    # 检查报告文件是否存在
    REPORT_FILE="${NSYS_OUTPUT}.nsys-rep"
    if [ ! -f "$REPORT_FILE" ]; then
        echo "❌ Error: Expected report file '$REPORT_FILE' was not found after successful profiling." >&2
        exit 1
    fi

    # 生成 CSV 报告
    echo "=== [Step 2] 开始 Nsight Systems (nsys) 统计导出 ==="
    # 注意：输入文件后缀必须是 .nsys-rep (nsys 默认生成的格式)
    nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum \
    --format csv \
    --output ${NSYS_OUTPUT} \
    "$REPORT_FILE"

    echo "==============================================================="
    echo "✅ nsys CSV reports exported to: ${OUTPUT_DIR}/"
    echo "   Files: $(ls ${OUTPUT_DIR}/*.csv 2>/dev/null || echo 'none')"

else
    # 如果 nsys profile 失败，同样将临时日志复制出来供检查
    cat "$TEMP_LOG" | tee "${OUTPUT_DIR}/nsys_profile.log"
    echo "❌ Error: nsys profile command failed. Check the log at ${OUTPUT_DIR}/nsys_profile.log" >&2
    exit 1
fi

# 清理临时文件
rm -f "$TEMP_LOG"



echo "=== [Step 3] 开始 Nsight Compute (ncu) 采集 ==="

NCU_LOG="${OUTPUT_DIR}/ncu_profile.log"

if ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --set speedOfLight \
  --force-overwrite \
  --export ${NCU_OUTPUT} \
  ${GRAPH_CMD} > "$NCU_LOG" 2>&1; then

    echo "✅ ncu profile completed successfully."

    NCU_REPORT="${NCU_OUTPUT}.ncu-rep"

    if [ ! -f "$NCU_REPORT" ]; then
        echo "❌ Error: Expected report file '$NCU_REPORT' not found." >&2
        exit 1
    fi

    echo "=== [Step 4] 导出 Nsight Compute CSV ==="

    ncu --import "$NCU_REPORT" \
        --csv \
        --page raw \
        > "${NCU_OUTPUT}.csv"

    echo "==============================================================="
    echo "✅ ncu CSV exported to: ${NCU_OUTPUT}.csv"

else
    echo "❌ Error: ncu profile command failed. Check log: $NCU_LOG" >&2
    exit 1
fi

