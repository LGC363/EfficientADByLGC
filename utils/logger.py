import logging
import os

# 配置日志记录
def setup_logger(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 如果日志目录不存在，则创建

    # 设置日志格式和日志级别
    logger = logging.getLogger("EfficientADLogger")
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    log_file = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

