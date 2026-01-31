import torch
import torch.utils.cpp_extension

# 查看 PyTorch 编译时使用的 CUDA 路径
print(f"CUDA Home: {torch.utils.cpp_extension.CUDA_HOME}")

# 查看底层具体的共享库位置
import os
import torch
torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
print(f"Torch Lib Path: {torch_lib_path}")

# 查看当前进程加载的 libcuda.so (驱动库) 和 libcudart.so (运行库)
os.system(f"grep 'libcuda' /proc/{os.getpid()}/maps")
