# Tensorflow Graph Optim
- 优化 tensorflow graph，尤其是客户 release 的 .pb / .spec 等

```bash
# 将 spec 文件转成 pb 并优化
python convert_spec_to_frozen_graph_def.py

# 推理 pb 模型（修改runner脚本里的musa_plugin 路径）
python graph_runner.py
```