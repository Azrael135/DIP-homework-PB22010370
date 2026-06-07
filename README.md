# DIP-homework-PB22010370
This is the assignment for the digital image processing course, PB22010370.
## 1. Introduction

本实验实现了一个简化版 3D Gaussian Splatting（3DGS）重建与渲染流程。给定多视角图像，首先使用 COLMAP 恢复相机内外参与稀疏三维点云，然后将每个 COLMAP 三维点初始化为一个可学习的 3D Gaussian，并通过可微分投影与 alpha-blending 完成图像重建。

本实验主要包含三个部分：

1. 使用 COLMAP 完成 Structure-from-Motion，获得相机参数和稀疏点云；
2. 使用纯 PyTorch 实现简化版 3DGS，包括 Gaussian 初始化、3D 到 2D 投影、Gaussian rasterization 和 alpha-blending；
3. 使用官方 3DGS 实现运行相同数据集，并从渲染质量、训练速度和显存占用三个方面与简化版实现进行对比。



## 2. Requirements

### 2.1 Hardware

实验使用的硬件环境如下：

| Item | Description |
|||
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
### 5.1 Gaussian Parameterization

在简化版 3DGS 中，每个 COLMAP 三维点被初始化为一个 3D Gaussian。每个 Gaussian 包含如下可学习参数：

| Parameter | Meaning |
|||
| Position `μ` | 三维空间中的中心位置 |
| Color `c` | RGB 颜色 |
| Opacity `o` | 不透明度 |
| Scale `S` | 三维尺度 |
| Rotation `R` | 由四元数表示的旋转 |

代码中使用 COLMAP 点云初始化 Gaussian 的位置和颜色，并使用局部邻域距离初始化尺度。旋转参数初始化为单位四元数，不透明度初始化为较高值。

3D 协方差矩阵由旋转矩阵和尺度矩阵构造：

```math
\Sigma = R S S^T R^T
```

该公式保证协方差矩阵是半正定的，并且可以通过优化尺度和旋转控制 Gaussian 的形状和方向。

### 5.2 Projection from 3D to 2D

对于每个 3D Gaussian，首先根据相机外参将其从世界坐标系变换到相机坐标系，然后利用相机内参投影到图像平面。为了将三维协方差投影为二维协方差，使用投影函数的一阶雅可比矩阵：

```math
\Sigma' = J W \Sigma W^T J^T
```

其中，`W` 表示世界坐标到相机坐标的旋转变换，`J` 是透视投影的雅可比矩阵，`\Sigma'` 是投影到图像平面后的二维协方差。



### 5.3 2D Gaussian Rasterization

投影后，每个 Gaussian 在图像平面上形成一个二维 Gaussian。对于像素位置 `x`，二维 Gaussian 的值为：

```math
f(x; \mu_i, \Sigma_i)
=
\frac{1}{2\pi\sqrt{|\Sigma_i|}}
\exp
\left(
-\frac{1}{2}(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)
\right)
```

该值表示当前 Gaussian 对该像素的影响强度。



### 5.4 Alpha Blending

为了得到最终颜色，需要按照深度顺序对所有 Gaussian 进行 alpha-blending。每个 Gaussian 在像素处的 alpha 定义为：

```math
\alpha_i(x) = o_i f(x; \mu_i, \Sigma_i)
```

透射率为：

```math
T_i(x) = \prod_{j<i} (1-\alpha_j(x))
```

最终像素颜色为：

```math
C(x) = \sum_i T_i(x)\alpha_i(x)c_i
```



### 5.5 Training

训练命令如下：

```bash
python train.py \
  --colmap_dir data/chair \
  --checkpoint_dir data/chair/checkpoints \
  --num_epochs 200 \
  --debug_every 5 \
  --device cuda
```

为了记录训练时间，可以使用：

```bash
/usr/bin/time -v python train.py \
  --colmap_dir data/chair \
  --checkpoint_dir data/chair/checkpoints \
  --num_epochs 200 \
  --debug_every 5 \
  --device cuda \
  2>&1 | tee data/chair/checkpoints/simple_train.log
```

为了记录显存占用，在另一个终端运行：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu \
  --format=csv -l 1 \
  > data/chair/checkpoints/simple_nvidia_smi.csv
```



### 5.6 Results

简化版 3DGS 的训练结果保存在：

```text
data/chair/checkpoints/
├── checkpoint_000000.pt
├── checkpoint_000020.pt
├── checkpoint_000040.pt
├── ...
├── debug_images/
└── debug_rendering.mp4
```

训练过程中保存的 debug 图像可以用于观察重建质量变化。下图展示了训练过程中的 GT 图像与 rendered 图像对比：
<img width="800" height="800" alt="00012" src="https://github.com/user-attachments/assets/bb6adb01-ad7c-42c1-a936-e33fb9581a3b" />
<img width="800" height="800" alt="00010" src="https://github.com/user-attachments/assets/18e05c61-45df-4edb-8752-884216eac27e" />
<img width="800" height="800" alt="00008" src="https://github.com/user-attachments/assets/4ba9321d-f41d-470f-8f7e-3d9d32702395" />
<img width="800" height="800" alt="00005" src="https://github.com/user-attachments/assets/da4cc528-6fdc-4b0d-86dd-b54d490fa091" />
<img width="800" height="800" alt="00002" src="https://github.com/user-attachments/assets/103a3e52-f177-4600-8a90-5ac5a7123b7d" />


### 5.7 Discussion

简化版 PyTorch 3DGS 能够学习出物体的大致形状和颜色分布，但渲染结果仍然存在模糊、空洞和边界不清晰等问题。主要原因包括：

1. 只使用 COLMAP 稀疏点作为初始 Gaussian，点数较少；
2. 没有实现 adaptive Gaussian densification，无法在细节区域动态增加 Gaussian；
3. 使用纯 PyTorch 在整张图像上显式计算 `(N, H, W)` 级别的 Gaussian 值，计算和显存开销较大；
4. 没有使用 tile-based rasterization，因此大量计算发生在 Gaussian 实际影响区域之外。



## 6. Task 3: Comparison with Official 3DGS

### 6.1 Official 3DGS Training

为了与简化版实现进行对比，本实验使用官方 3DGS 实现在相同的 `chair` 数据集上进行训练。

进入官方仓库：

```bash
cd /path/to/gaussian-splatting
conda activate gaussian_splatting
```

运行官方训练：

```bash
/usr/bin/time -v python train.py \
  -s /path/to/assignment4_3dgs/data/chair \
  -m /path/to/assignment4_3dgs/official_outputs/chair \
  --eval \
  --iterations 30000 \
  2>&1 | tee /path/to/assignment4_3dgs/official_outputs/chair_train.log
```

显存记录命令：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu \
  --format=csv -l 1 \
  > /path/to/assignment4_3dgs/official_outputs/chair_nvidia_smi.csv
```



### 6.2 Official Rendering

训练完成后，使用官方渲染脚本生成测试视角结果：

```bash
python render.py \
  -m /path/to/assignment4_3dgs/official_outputs/chair
```



### 6.3 Official Metrics

使用官方评估脚本计算 PSNR、SSIM 和 LPIPS：

```bash
python metrics.py \
  -m /path/to/assignment4_3dgs/official_outputs/chair
```



### 6.4 Quantitative Comparison

| Method | Training Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time | Peak GPU Memory |
||:|:|:|:|:|:|
| Simplified PyTorch 3DGS | 200 epochs | - | - | - | xxx min | xxx MB |
| Official 3DGS | 30000 iterations | xxx | xxx | xxx | xxx min | xxx MB |

说明：

- `Training Time` 从 `/usr/bin/time -v` 输出中的 `Elapsed (wall clock) time` 读取；
- `Peak GPU Memory` 从 `nvidia-smi` 记录的最大 `memory.used` 读取；
- 官方 3DGS 的 PSNR / SSIM / LPIPS 来自 `metrics.py`；
- 简化版如果没有实现单独测试集指标，可以主要使用可视化结果和训练 loss 进行定性比较。



### 6.5 Qualitative Comparison

简化版 3DGS 结果：



官方 3DGS 结果：

<img width="800" height="800" alt="00008" src="https://github.com/user-attachments/assets/69af9b35-5328-485f-ad51-3da9a78699ee" />
<img width="800" height="800" alt="00007" src="https://github.com/user-attachments/assets/35e0e1d8-4d74-4e66-a5e3-41163948cb87" />
<img width="800" height="800" alt="00012" src="https://github.com/user-attachments/assets/ef2e292d-8b95-4011-967d-d42bd3124c0b" />
<img width="800" height="800" alt="00001" src="https://github.com/user-attachments/assets/faaf6168-bb04-48e9-ae50-f5334b5d0b39" />




从可视化结果可以观察到，官方 3DGS 的渲染质量明显更高。物体边界更加清晰，纹理细节更加稳定，多视角一致性也更好。相比之下，简化版 PyTorch 实现能够恢复物体的大致形状，但图像整体较模糊，并且局部区域可能出现空洞或颜色不稳定。



## 7. Analysis

### 7.1 Rendering Quality

官方 3DGS 的渲染质量优于简化版，主要原因是官方实现包含 adaptive Gaussian densification。该机制可以在训练过程中根据梯度和重建误差动态增加 Gaussian 数量，使物体边缘、纹理和高频细节区域得到更充分的表示。

简化版实现只使用 COLMAP 的稀疏点作为初始 Gaussian，Gaussian 数量固定。因此，当初始点云较稀疏时，模型难以覆盖整个物体表面，导致渲染结果出现模糊或空洞。



### 7.2 Training Speed

官方 3DGS 训练速度更快，主要原因是其使用了专门设计的 CUDA rasterizer 和 tile-based rendering。该实现只在 Gaussian 可能影响的局部区域进行计算，从而避免了大量无效像素计算。

简化版 PyTorch 实现中，每个 Gaussian 都需要在整张图像网格上计算二维 Gaussian 值，形成形如 `(N, H, W)` 的中间张量。当 Gaussian 数量或图像分辨率增大时，计算量会迅速上升。



### 7.3 GPU Memory Usage

简化版实现的显存开销主要来自显式构造大规模中间张量。例如，在计算二维 Gaussian value 时，需要保存每个 Gaussian 对每个像素的影响，因此显存复杂度近似与 `N × H × W` 成正比。

官方实现通过 CUDA kernel 和 tile-based rasterization 避免了完整存储所有 Gaussian 对所有像素的影响，因此显存利用更加高效。同时，官方实现还包含 visibility-aware rendering，只处理对当前视角有贡献的 Gaussian。



### 7.4 Implementation Difference

| Component | Simplified PyTorch 3DGS | Official 3DGS |
||||
| Gaussian Initialization | COLMAP sparse points | COLMAP sparse points |
| Gaussian Number | Fixed | Adaptive densification |
| Rasterization | Pure PyTorch dense computation | CUDA tile-based rasterizer |
| Optimization | Basic RGB/L1 loss | More complete 3DGS optimization |
| Rendering Efficiency | Low | High |
| Quality | Coarse reconstruction | High-quality novel view synthesis |



## 8. Reproducibility

完整复现实验可以按照以下顺序运行。

### Step 1: COLMAP reconstruction

```bash
python mvs_with_colmap.py --data_dir data/chair
```

### Step 2: Reprojection check

```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

### Step 3: Train simplified PyTorch 3DGS

```bash
python train.py \
  --colmap_dir data/chair \
  --checkpoint_dir data/chair/checkpoints \
  --num_epochs 200 \
  --debug_every 5 \
  --device cuda
```

### Step 4: Render simplified 3DGS video

```bash
python render_3dgs_mv.py \
  --colmap_dir data/chair \
  --checkpoint data/chair/checkpoints/checkpoint_000180.pt \
  --output data/chair/render_mv.mp4 \
  --num_frames 240 \
  --fps 30
```

### Step 5: Train official 3DGS

```bash
cd /path/to/gaussian-splatting

python train.py \
  -s /path/to/assignment4_3dgs/data/chair \
  -m /path/to/assignment4_3dgs/official_outputs/chair \
  --eval \
  --iterations 30000
```

### Step 6: Render official 3DGS

```bash
python render.py \
  -m /path/to/assignment4_3dgs/official_outputs/chair
```

### Step 7: Evaluate official 3DGS

```bash
python metrics.py \
  -m /path/to/assignment4_3dgs/official_outputs/chair
```



## 9. Conclusion

本实验完成了一个简化版 3D Gaussian Splatting 系统，并将其与官方 3DGS 实现进行了对比。通过 Task 1，实验使用 COLMAP 从多视角图像中恢复了相机参数和稀疏三维点云；通过 Task 2，实验基于 PyTorch 实现了 3D Gaussian 初始化、协方差投影、二维 Gaussian 计算和 alpha-blending 渲染；通过 Task 3，实验进一步比较了简化实现与官方实现的渲染质量、训练速度和显存占用。

实验结果表明，简化版实现有助于理解 3DGS 的核心数学原理和渲染流程，但由于缺少 adaptive densification、CUDA rasterizer 和 tile-based rendering，其渲染质量和运行效率均明显弱于官方实现。官方 3DGS 更适合实际高质量新视角合成任务，而简化版实现更适合作为理解 3DGS 基础机制的教学版本。



## 10. References

- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering.
- Official 3DGS implementation: https://github.com/graphdeco-inria/gaussian-splatting
- Research code README template: https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md
- COLMAP: https://colmap.github.io/

