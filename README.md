# EfficientADByLGC

EfficientADByLGC 是一个轻量高效的工业图像异常检测复现项目，基于论文 [EfficientAD](https://arxiv.org/abs/2303.14535)。本项目旨在实现并优化论文中的技术，包括轻量化特征提取器、学生-教师模型和逻辑异常检测。
---

## 项目结构

以下是项目的目录结构，以树状图形式展示各目录和文件的功能：

EfficientAD/ ├── data/ # 数据集存放目录 │ ├── mvtec_ad/ # MVTec AD 数据集目录 │ ├── visa/ # VisA 数据集目录 │ └── loco/ # LOCO 数据集目录 │ ├── models/ # 模型代码 │ ├── feature_extractor.py # 轻量级特征提取器（Patch Description Network） │ ├── student_teacher.py # 学生-教师网络 │ └── autoencoder.py # 自动编码器模型 │ ├── utils/ # 工具代码 │ ├── dataset.py # 数据加载和预处理 │ ├── visualizer.py # 可视化工具 │ ├── metrics.py # 性能评估指标（如 AU-ROC、AU-PRO） │ └── logger.py # 日志记录工具 │ ├── experiments/ # 实验主代码 │ ├── train.py # 训练主程序 │ ├── eval.py # 模型评估程序 │ └── inference.py # 单张图片推理脚本 │ ├── logs/ # 训练和评估日志 │ └── train.log # 自动生成的训练日志 │ ├── checkpoints/ # 模型保存目录 │ ├── best_model.pth # 最优模型权重 │ └── latest_model.pth # 最新保存的模型权重 │ ├── requirements.txt # Python 依赖包列表 └── README.md # 项目介绍文档
---

## 使用说明

### 1. 环境配置
- 推荐使用 **Python 3.9**。
- 安装依赖：
  ```bash
  pip install -r requirements.txt
