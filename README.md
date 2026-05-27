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
| GPU | NVIDIA A100 / A6000 / RTX xxxx |
| CPU | xxx |
| Memory | xxx GB |
| OS | Ubuntu 22.04 |
