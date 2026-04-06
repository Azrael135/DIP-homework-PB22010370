# DIP-homework-PB22010370
This is the assignment for the digital image processing course, PB22010370.
## data_poisson结果


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


