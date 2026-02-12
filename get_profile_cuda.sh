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


echo "=== [Step 0] 目录准备完成: ${OUTPUT_DIR} ==="

# 1) 定义要运行的图形/训练命令
# 请确保 python 路径和脚本路径正确
GRAPH_CMD="python ./graph_runner_profile.py --graph ${GRAPH_PATH} --output ${OUTPUT_DIR} --platform cuda"

echo "=== [Step 1] 开始 Nsight Systems (nsys) 采集 ==="
# 注意：nsys profile 成功后会生成 .nsys-rep 文件
# 如果 graph_runner.py 运行时间很长，nsys 会一直运行直到它结束
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=${NSYS_OUTPUT} \
  ${GRAPH_CMD} 2>&1 | tee "${OUTPUT_DIR}/nsys_profile.log"

if [ $? -ne 0 ]; then
    echo "ERROR: nsys profile 失败，停止后续步骤。"
    exit 1
fi

echo "=== [Step 2] 开始 Nsight Systems (nsys) 统计导出 ==="
# 注意：输入文件后缀必须是 .nsys-rep (nsys 默认生成的格式)
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum \
  --format csv \
  --output ${NSYS_OUTPUT} \
  ${NSYS_OUTPUT}.nsys-rep

if [ $? -ne 0 ]; then
    echo "WARNING: nsys stats 执行出错，但尝试继续。"
fi

