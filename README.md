# Tensorflow Graph Optim
- 优化 tensorflow graph，尤其是客户 release 的 .pb / .spec 等

```bash
# 将 spec 文件转成 pb 并优化
python convert_spec_to_frozen_graph_def.py


# 1) MUSA推理pb模型：固定随机输入并保存输入+输出
python graph_runner.py \
  --spec-path meta_graph_1.spec \
  --pb-path meta_graph_1_frozen.pb \
  --seed 2026 \
  --save-input-npz inputs_fixed.npz \
  --save-output-npz musa_outputs.npz \
  --load-musa-plugin

# 2) CUDA侧推理pb模型：加载同一份输入，保存输出
python graph_runner.py \
  --spec-path meta_graph_1.spec \
  --pb-path meta_graph_1_frozen.pb \
  --load-input-npz inputs_fixed.npz \
  --save-output-npz cuda_outputs.npz

# 3) 对比输出
python compare_musa_cuda_outputs.py \
  --musa-output musa_outputs.npz \
  --cuda-output cuda_outputs.npz \
  --atol 1e-5 \
  --rtol 1e-4 \
  --topk 20
```