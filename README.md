# DIP-homework-PB22010370
This is the assignment for the digital image processing course, PB22010370.
## 基于 PyTorch 的 Poisson Image Editing 实验报告
本实验旨在实现一个基于 PyTorch 的 Poisson Image Editing 系统，并通过交互式界面完成前景区域到背景图像的无缝融合。

具体目标包括：

1. 实现 Polygon to Mask 功能，将用户在图像上选取的多边形区域转换为二值掩码。
2. 实现 Laplacian Distance Computation，利用拉普拉斯算子约束融合区域的梯度结构。
3. 基于优化的方法，在背景图像指定区域内重建融合结果，使前景目标能够较自然地嵌入背景中。
### 1. 实验原理
Poisson Image Editing 的核心思想是：

不直接复制像素值，而是尽可能保持源图像在选定区域内的梯度场，同时使融合结果与目标背景在边界处自然衔接。

设源图像为前景图像，目标图像为背景图像，目标是在背景的某个区域中重建一张新图，使其内部梯度尽量接近源图像，而边界与背景一致。

在离散图像上，这一问题通常可转化为对拉普拉斯约束的优化问题。实验中使用了 4 邻域 Laplacian kernel：

[ 0  -1   0
 -1   4  -1
  0  -1   0 ]

通过比较前景图像和融合图像在区域内的拉普拉斯响应，可以度量两者局部结构的一致性。

### 2. 实现方法
#### 2.1 多边形区域选择

程序允许用户在前景图像上点击多个点，逐步构建一个多边形区域。
当点数不少于 3 个时，点击 “Close Polygon” 即可闭合多边形。程序会将该多边形作为待融合区域。

#### 2.2 Polygon to Mask

在 create_mask_from_points 函数中，首先创建一个与图像大小一致的全零矩阵，然后利用 PIL.ImageDraw.Draw.polygon() 将多边形内部填充为 255，最终得到二值 mask。

该方法的优点是：

实现简单直接；与用户交互的多边形点坐标天然匹配；可以方便地得到前景区域 mask 和平移后的背景区域 mask
#### 2.3 Laplacian Distance Computation

在 cal_laplacian_loss 中，程序分别对前景图像和当前融合图像进行卷积计算，得到各自的拉普拉斯响应。
随后，仅在掩码区域内计算二者的平方误差，并对像素数进行归一化，构成最终 loss。

其作用是：

保持源区域的结构信息
避免单纯像素拷贝造成明显边界
通过优化逐步逼近更自然的融合效果
#### 2.4 优化融合过程

在 blending 函数中，程序首先：

1. 将前景图像与背景图像转为 tensor
2. 根据前景 polygon 点和位移 (dx, dy) 构造前景 mask 与背景 mask
3. 用背景图像初始化待优化的融合图像 blended_img

然后通过 Adam 优化器迭代更新融合区域，只在背景 mask 对应的位置进行优化，其他区域保持背景图像不变。损失函数为 Laplacian loss。

实验中迭代步数设置为 5000，学习率初始为 1e-2，并在后期衰减。

### 3. 代码结构说明

本实验的主要模块包括：

initialize_polygon()：初始化多边形状态
add_point()：响应鼠标点击，添加多边形顶点
close_polygon()：闭合多边形
create_mask_from_points()：根据多边形点生成 mask
cal_laplacian_loss()：计算拉普拉斯损失
blending()：执行 Poisson 融合优化
Gradio 界面部分：用于实现前景/背景上传、区域选择、位置调整与结果显示
### 4. 实验结果分析
#### equation
<img width="1308" height="1048" alt="image" src="https://github.com/user-attachments/assets/55afa70e-e313-420a-b7f9-b74c7d5f02de" />

#### monolisa
<img width="2748" height="2136" alt="image" src="https://github.com/user-attachments/assets/495ee9ae-8543-43d4-9df3-9a2bbd310169" />

#### water


本实验成功实现了一个交互式的 Poisson Image Editing 系统，能够完成以下流程：

1. 用户上传前景图像和背景图像
2. 在前景图像中圈定待融合区域
3. 调整该区域在背景图中的偏移位置
4. 自动执行优化并输出融合结果

从结果上看：

优点：能较好保留前景区域的结构信息；边界过渡比直接复制粘贴更加自然；支持交互式位置调整，使用方便
不足：当前实现只使用 Laplacian loss，结果仍可能存在颜色不完全协调的问题；优化迭代较多，运行速度偏慢，CPU 下尤其明显；当前景与背景差异较大时，融合结果仍可能显得不够真实
### 5. 问题与改进方向

本实验已经完成作业要求，但仍有一些可以进一步改进的方向：

1. 增加边界约束：除了梯度一致性，还可以显式约束边缘区域与背景更加平滑地衔接。
2. 加入颜色一致性损失：仅保持梯度可能导致颜色漂移，可额外加入像素级重建项。
3. 减少优化时间：目前迭代 5000 次，耗时较长，可以考虑更高效的数值求解方法。
4. 更严格的区域映射：当前是通过平移 (dx, dy) 将前景 polygon 映射到背景，后续可扩展为缩放、旋转等更灵活的变换。
### 6. 实验结论
实验结果表明，该方法能够通过优化保持前景区域的局部结构，并在背景图像中实现较自然的融合。虽然当前实现仍存在颜色协调和运行效率方面的不足，但整体上已经验证了 Poisson Image Editing 的基本思想与可行性。

### 7. 附：核心实现说明
（1）Polygon to Mask
使用 PIL.ImageDraw 将多边形区域填充为二值掩码，作为融合区域的空间约束。

（2）Laplacian Loss
使用 4 邻域 Laplacian kernel 对前景图像与融合图像分别求响应，仅在 mask 区域内计算均方误差。

（3）Blending Optimization
将融合图像设为可优化变量，使用 Adam 迭代更新，在目标区域内最小化 Laplacian loss。

## 基于全卷积网络的 Pix2Pix 实现
### 1. 任务描述

本实验实现了一个基于 全卷积网络（Fully Convolutional Network, FCN） 的图像到图像翻译模型（Pix2Pix 简化版本）。

任务目标为：
将输入的 语义分割图（semantic map） 转换为对应的 真实建筑图像（RGB image）。

数据集使用 Facades Dataset。下载方法：windows电脑安装WSL后，执行命令bash download_facades_dataset.sh，下载dataset。

### 2. 方法介绍

本实验采用 Encoder–Decoder 结构的全卷积网络：

模型结构：Fully Convolutional Network
输入：语义标签图（3通道）
输出：生成的建筑图像（3通道）
训练设置：
损失函数：L1 Loss
优化器：Adam
学习率：0.001
Betas：(0.5, 0.999)
训练轮数：300 epochs
学习率衰减：StepLR

### 3. 实验结果
可视化结果

每一行表示：

[真实图像 | 语义标签图 | 生成结果]
#### 示例1
<img width="768" height="256" alt="result_1" src="https://github.com/user-attachments/assets/a7c2e607-589a-47d1-9329-7ef1ff2f6330" />

#### 示例2
<img width="768" height="256" alt="result_2" src="https://github.com/user-attachments/assets/4bc6fbb3-17ff-441b-8a22-03c2eb1f28e3" />

#### 示例3
<img width="768" height="256" alt="result_3" src="https://github.com/user-attachments/assets/93d5be21-2f3e-47d0-9983-107200dbbfc7" />

#### 示例4
<img width="768" height="256" alt="result_4" src="https://github.com/user-attachments/assets/cbff57ad-c15a-46b7-b22a-47d715e19da8" />

#### 示例5
<img width="768" height="256" alt="result_5" src="https://github.com/user-attachments/assets/e1cf0535-06a5-401a-9626-420c7b0b679f" />

完整的训练过程和结果:
train_results.zip: https://pan.ustc.edu.cn/share/index/a1ee8db8ffd24333a3a0?p=1 密码：123456
var_results.zip: https://pan.ustc.edu.cn/share/index/562d282f77fb4a9293ad?p=1 密码：123456

### 4. 训练效果
最终训练集损失（Train Loss）：≈ 0.23
最终验证集损失（Validation Loss）：≈ 0.40

模型在训练过程中收敛稳定，没有出现震荡或发散。

### 5. 结果分析

从生成结果可以观察到：

（1）优点
模型能够学习到：建筑整体结构+窗户排列规律+层级分布信息，输出图像在空间布局上基本正确
（2）不足
图像较为模糊、缺乏纹理细节、局部区域存在颜色不稳定
（3）原因分析
仅使用 L1 损失；倾向于生成“平均结果”（导致模糊）；未使用 对抗训练（GAN）；缺乏对真实感的约束；数据集规模较小；泛化能力有限。

### 6. 讨论
Facades 数据集规模较小，限制了模型的表达能力。
若要进一步提升效果，可以：
引入 GAN 判别器（完整 Pix2Pix）；使用更大规模数据集；加入 perceptual loss 或 feature loss

### 7. 实验结论
本实验验证了：全卷积网络可以完成基本的图像翻译任务，但仅依赖 L1 loss 难以生成高质量图像
因此，在 Pix2Pix 框架中，引入对抗学习是提升生成质量的关键。


