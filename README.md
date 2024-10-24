# 多层感知器（MLP）实现

本项目提供了使用 **CPU（NumPy）** 和 **GPU（CuPy）** 的两种版本的多层感知器（MLP）。其中，GPU 版本需要对应的 CUDA 驱动支持。

## 文件内容

- `cpu_version.py`：使用 NumPy 实现的 CPU 版本。
- `gpu_version.py`：使用 CuPy 实现的 GPU 版本。
- `mlp_weights98.5.npz`：已训练完成的权重文件。

## 前置要求

- Python 3.x
- NumPy 库
- CuPy 库（仅 GPU 版本需要）
- NVIDIA GPU 硬件（仅 GPU 版本需要）
- 对应版本的 CUDA 驱动

## CUDA 驱动安装指南

1. **检查 CUDA 版本**

   打开命令提示符，输入以下命令：`nvidia-smi`
   
   这将显示您当前的 CUDA 版本。

2. **下载对应的 CUDA 驱动**

根据查询到的 CUDA 版本，前往 [NVIDIA 官方网站](https://developer.nvidia.com/cuda-downloads) 下载相应版本的 CUDA 驱动（一般情况下可选择默认配置）。

3. **安装并重启**

运行下载的安装程序，完成后重启计算机。

4. **设置环境变量**

安装完成后，CUDA 的路径通常会自动添加到环境变量中。如果没有，请手动添加 CUDA 的安装路径到系统的环境变量。

## 运行指南

### CPU 版本

1. **安装依赖**
`pip install numpy`

3. **运行程序**
python cpu_version.py


### GPU 版本

1. **安装依赖**
`pip install cupy`

2. **确保 CUDA 驱动已正确安装并配置**

参考上方的 CUDA 驱动安装指南。

3. **运行程序**
`python gpu_version.py`

## 使用预训练权重

程序将自动加载 `mlp_weights98.5.npz` 文件中的已训练权重，达到更好的初始性能。

## 注意事项

- 确保您的 GPU 兼容所安装的 CUDA 版本。
- 如果在运行 GPU 版本时遇到问题，请检查环境变量和 CuPy 的安装是否正确。



