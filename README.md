# EfficientADByLGC

EfficientADByLGC 是一个轻量高效的工业图像异常检测复现项目，基于论文 [EfficientAD](https://arxiv.org/abs/2303.14535)。本项目旨在实现并优化论文中的技术，包括轻量化特征提取器、学生-教师模型和逻辑异常检测。
```

## 项目结构

以下是项目的目录结构：

- EfficientAD/
  - data/                      # 数据集存放目录
    - mvtec_ad/                # MVTec AD 数据集目录
    - visa/                    # VisA 数据集目录
    - loco/                    # LOCO 数据集目录
  - models/                    # 模型代码
    - feature_extractor.py     # 轻量级特征提取器（Patch Description Network）
    - student_teacher.py       # 学生-教师网络
    - autoencoder.py           # 自动编码器模型
  - utils/                     # 工具代码
    - dataset.py               # 数据加载和预处理
    - visualizer.py            # 可视化工具
    - metrics.py               # 性能评估指标（如 AU-ROC、AU-PRO）
    - logger.py                # 日志记录工具
  - experiments/               # 实验主代码
    - train.py                 # 训练主程序
    - eval.py                  # 模型评估程序
    - inference.py             # 单张图片推理脚本
  - logs/                      # 训练和评估日志
    - train.log                # 自动生成的训练日志
  - checkpoints/               # 模型保存目录
    - best_model.pth           # 最优模型权重
    - latest_model.pth         # 最新保存的模型权重
  - requirements.txt           # Python 依赖包列表
  - README.md                  # 项目介绍文档

```

## 使用说明

### 1. 环境配置

为确保项目在本地能够顺利运行，以下是详细的环境配置步骤：

#### 步骤 1：安装 Python

- 本项目推荐使用 **Python 3.9**。你可以从 [Python 官网](https://www.python.org/downloads/) 下载并安装 Python。

- 安装后，确认 Python 和 `pip` 的版本：
  ```bash
  python --version
  pip --version
  ```

#### 步骤 2：创建虚拟环境（可选）

为了避免依赖包冲突，推荐使用虚拟环境。可以通过 `conda` 来创建虚拟环境。

- **使用 Anaconda 创建虚拟环境**：
  ```bash
  conda create -n efficientad python=3.9
  conda activate efficientad
  ```

#### 步骤 3：安装依赖包

在激活的虚拟环境中，使用以下命令安装项目所需的所有 Python 包：

```bash
pip install -r requirements.txt
```

将自动从 `requirements.txt` 中安装以下依赖：
- `torch`
- `torchvision`
- `matplotlib`
- `opencv-python`
- `scikit-learn`
- `scikit-image`
- `albumentations`
- `tqdm`

#### 步骤 4：配置数据集

你需要下载并配置 MVTec AD、VisA 和 LOCO 数据集。这些数据集用于训练和评估模型。

- **MVTec AD 数据集**：[MVTec AD 数据集](https://www.mvtec.com/company/research/datasets/mvtec-ad)
  - 注册并同意许可协议后，下载数据集。
  - 解压数据集并将其存放在 `data/mvtec_ad/` 目录下。
  - 示例路径：`data/mvtec_ad/screw/`

- **VisA 数据集**：[VisA 数据集](https://github.com/mauricius/VisA)
  - 下载并将其存放在 `data/visa/` 目录下。

- **LOCO 数据集**：[LOCO 数据集](https://github.com/Edouard-Legendre/LOCO)
  - 下载并将其存放在 `data/loco/` 目录下。

确保将数据集按照上述目录结构放置，方便后续代码加载和处理。

#### 步骤 5：验证环境配置

安装依赖和配置数据集后，可以验证是否成功配置环境。首先，运行以下命令来测试 Python 环境和依赖包是否安装正确：

```bash
python -c "import torch; print(torch.__version__)"
```

如果没有错误并正确输出 PyTorch 版本号，则说明环境配置成功。

---

### 2. 运行项目

#### 步骤 1：训练模型

使用以下命令开始训练模型：

```bash
python experiments/train.py
```

该命令将启动训练过程，训练日志会输出到 `logs/train.log` 文件中，并且在控制台实时显示。

#### 步骤 2：评估模型

训练完成后，可以使用以下命令评估模型：

```bash
python experiments/eval.py
```

该命令会在测试集上评估模型的性能，输出模型的评估指标（如 AU-ROC、AU-PRO 等）。

#### 步骤 3：推理单张图片

如果你想对单张图像进行推理，使用以下命令：

```bash
python experiments/inference.py --image <path_to_image>
```

将 `<path_to_image>` 替换为你想要推理的图像路径。

---