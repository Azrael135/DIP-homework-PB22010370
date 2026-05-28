# DIP-homework-PB22010370
This is the assignment for the digital image processing course, PB22010370.
## 1. Introduction

本实验实现了一个简化版 3D Gaussian Splatting（3DGS）重建与渲染流程。给定多视角图像，首先使用 COLMAP 恢复相机内外参与稀疏三维点云，然后将每个 COLMAP 三维点初始化为一个可学习的 3D Gaussian，并通过可微分投影与 alpha-blending 完成图像重建。

本实验主要包含三个部分：

1. 使用 COLMAP 完成 Structure-from-Motion，获得相机参数和稀疏点云；
2. 使用纯 PyTorch 实现简化版 3DGS，包括 Gaussian 初始化、3D 到 2D 投影、Gaussian rasterization 和 alpha-blending；
3. 使用官方 3DGS 实现运行相同数据集，并从渲染质量、训练速度和显存占用三个方面与简化版实现进行对比。

---

## 2. Requirements

### 2.1 Hardware

实验使用的硬件环境如下：

| Item | Description |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| OS | Ubuntu 22.04 |

### 2.2 Software

实验使用 conda 环境进行管理。

```bash
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
conda env create --file environment.yml
conda activate gaussian_splatting
```
主要依赖包括：
```bash
pip install numpy opencv-python tqdm natsort
pip install torch torchvision
pip install pytorch3d
```

因为这里gaussian_splatting，hanhai22超算平台的CUDA版本不匹配，需要额外安装:
```bash
conda install cudatoolkit cudatoolkit-dev -c conda-forge -y
```

安装colmap：
```bash
conda install -c conda-forge colmap -y
```
检查环境：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2, numpy, tqdm, natsort; print('basic packages ok')"
python -c "import pytorch3d; print('pytorch3d ok')"
colmap -h
```
# 3. Dataset

本实验使用 chair 数据集作为主要实验对象。数据目录结构如下：

data/
└── chair/
    ├── images/
    │   ├── 000.png
    │   ├── 001.png
    │   └── ...
    └── sparse/

其中 images/ 目录包含多视角 RGB 图像。实验首先使用 COLMAP 从这些多视角图像中恢复相机参数和稀疏点云。

# 4. Task 1: Structure-from-Motion with COLMAP
## 4.1 Method

首先使用 COLMAP 对多视角图像进行 Structure-from-Motion。该步骤包括特征提取、特征匹配、稀疏重建以及模型格式转换。得到的相机内参、相机外参和稀疏三维点将作为后续 3DGS 初始化的基础。

运行命令如下：

python mvs_with_colmap.py --data_dir data/chair

运行完成后，COLMAP 输出被保存到：

data/chair/sparse/0_text/
├── cameras.txt
├── images.txt
└── points3D.txt
## 4.2 Reprojection Check

为了检查 COLMAP 结果是否合理，将恢复出的 3D 点重新投影回原图：

python debug_mvs_by_projecting_pts.py --data_dir data/chair

投影结果保存在：

data/chair/projections/
## 4.3 Results

从重投影结果可以看到，大部分稀疏点能够落在物体表面附近，说明 COLMAP 恢复出的相机位姿和三维点云基本合理。因此，该稀疏点云可以作为后续 3D Gaussian 初始化的输入。

在报告中插入图片：


## 4.4 Discussion

COLMAP 输出的点云通常较为稀疏，能够描述物体的大致三维结构，但不足以直接生成高质量的稠密渲染结果。因此，在 Task 2 中，需要将这些稀疏点扩展为具有颜色、不透明度、尺度和旋转参数的 3D Gaussian，并通过优化逐步改善渲染质量。

# 5. Task 2: Simplified PyTorch 3D Gaussian Splatting
## 5.1 Gaussian Parameterization

在简化版 3DGS 中，每个 COLMAP 三维点被初始化为一个 3D Gaussian。每个 Gaussian 包含如下可学习参数：

