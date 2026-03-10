# DIP-homework-PB22010370
This is the assignment for the digital image processing course, PB22010370.
You can check my different branches to obtain my homework content.

## 环境配置

本项目在以下环境下测试通过：

- Python 3.10
- Conda environment: `dip_env`

创建并安装环境：

```bash
conda create -n dip_env python=3.10 -y
conda activate dip_env
pip install -r requirements.txt
```
### 运行：
To run basic transformation, run:

`python run_global_transform.py`

To run point guided transformation, run:

`python run_point_transform.py`

### 结果：
Basic Transformation：

![21526cdcd76150e60e4a99e395a127d9](https://github.com/user-attachments/assets/3487d2ef-ac69-466e-a292-1364f79f9313)

Point Guided Deformation:

<img width="994" height="1212" alt="image" src="https://github.com/user-attachments/assets/fc3183b4-9656-437b-a6d2-bc4da521f8ff" />

<img width="1288" height="1080" alt="image" src="https://github.com/user-attachments/assets/b0c9df09-0743-4a9d-9f24-f83dcb470315" />


备注：为了选取几个固定点，您可以前后两次**连续**点击同一个点，这样我们的向量位移为0，系统自动识别为固定点。

**应用：在设计妆容时，可以通过数字捏脸，确定眉型、脸型、眼型，从而塑造风格妆容。以鞠婧祎的图片为例，如果拉高眉尾，收缩颧骨，会加重冷感。
**
