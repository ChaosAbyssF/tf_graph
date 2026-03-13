# Tensorflow Graph Optim
- 优化 tensorflow graph，尤其是客户 release 的 .pb / .spec 等

```bash
# 将 spec 文件转成 pb 并优化
python convert_spec_to_frozen_graph_def.py --graph /path/to/graph.spec

# 推理 pb 模型（修改graph runner脚本里的musa_plugin 路径）  for latency test
python graph_runner.py --graph /path/to/graph.spec --batch_size 1024 --platfrom cpu/cuda/musa

# profiling pb 模型
bash get_getprofile_${platfrom}.sh --graph /path/to/graph.spec

# 寻找坏节点
# fix pb 文件
python scan_fix_frozen_pb.py --pb /path/to/graph.pb --fix
python detect_bad_node.py --graph /path/to/graph.spec --platform cpu/cuda/musa
```