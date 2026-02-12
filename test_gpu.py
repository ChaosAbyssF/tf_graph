import os
# 强制指定 CUDA 库路径 (根据 AutoDL L20 常见路径调整，通常是 /usr/local/cuda-12.x 或 /usr/local/cuda)
# 你可以先运行 `ls /usr/local/` 看看具体是哪个版本，比如 cuda-12.1
cuda_path = "/usr/local/cuda" 
if os.path.exists(cuda_path):
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import tensorflow as tf
print("TF Version:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
