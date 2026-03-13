#!/bin/bash

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

TEMP_LOG="${OUTPUT_DIR}/temp_profile.log"

echo "========================= running msys ============================"
echo "Model:       ${GRAPH_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "---------------------------------------------------------------"

# 先将 msys profile 的输出重定向到一个临时文件
if msys profile \
    --device=0 \
    --trace=musa,osrt \
    --output="${MSYS_OUTPUT}" \
    python graph_runner_profile.py --graph "${GRAPH_PATH}" \
    --output "${OUTPUT_DIR}" \
    --platform "musa" \
    > "$TEMP_LOG" 2>&1; then

    # 如果 msys profile 成功，再将临时日志文件的内容复制到最终日志，并打印到屏幕上
    cat "$TEMP_LOG" | tee "${OUTPUT_DIR}/msys_profile.log"
    echo "✅ msys profile completed successfully and log was written."
    
    # 检查报告文件是否存在
    REPORT_FILE="${MSYS_OUTPUT}.msys-rep"
    if [ ! -f "$REPORT_FILE" ]; then
        echo "❌ Error: Expected report file '$REPORT_FILE' was not found after successful profiling." >&2
        exit 1
    fi

    # 生成 CSV 报告
    echo "========================= generating msys stats ============================"
    msys stats \
        --format csv \
        --report musa_api_sum,musa_gpu_kern_sum,musa_kern_exec_sum,musa_api_gpu_sum \
        --output "${MSYS_OUTPUT}" \
        "$REPORT_FILE"

    echo "==============================================================="
    echo "✅ msys CSV reports exported to: ${OUTPUT_DIR}/"
    echo "   Files: $(ls ${OUTPUT_DIR}/*.csv 2>/dev/null || echo 'none')"

else
    # 如果 msys profile 失败，同样将临时日志复制出来供检查
    cat "$TEMP_LOG" | tee "${OUTPUT_DIR}/msys_profile.log"
    echo "❌ Error: msys profile command failed. Check the log at ${OUTPUT_DIR}/msys_profile.log" >&2
    exit 1
fi

# 清理临时文件
rm -f "$TEMP_LOG"

# 运行meta2的mcu需要将runner脚本的log关闭
echo "========================= running mcu ============================"

MCU_OUTPUT="${OUTPUT_DIR}/mcu_profile"
# 定义最终的日志文件
MCU_LOG="${OUTPUT_DIR}/mcu_profile.log"

# 使用管道组合命令：执行 -> 过滤 -> 输出到文件和屏幕
if mcu --devices 0 \
    --output ${MCU_OUTPUT} \
    --sections SpeedOfLight \
    --force-overwrite \
    python graph_runner_profile.py --graph "${GRAPH_PATH}" \
    --output "${OUTPUT_DIR}" \
    --platform "musa" \
    2>&1 | grep -v "^==PROF== Profiling" > "$MCU_LOG"; then

    # 命令成功后，将过滤后的日志内容打印到屏幕上
    # cat "$MCU_LOG"
    echo "✅ mcu profile completed successfully. Filtered log is at: $MCU_LOG"
    
else
    # 如果 mcu 失败，将过滤后的日志内容打印到错误输出
    echo "❌ Error: mcu command failed. Check the log at $MCU_LOG" >&2
    # 显示日志的最后几行以便快速诊断
    tail -n 20 "$MCU_LOG" >&2
    exit 1
fi
