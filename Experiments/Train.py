import torch
import torch.optim as optim
from models.student_teacher import StudentTeacherModel
from utils.dataset import load_data
from utils.logger import setup_logger

# 设置日志
logger = setup_logger()

def train():
    # 初始化数据
    train_loader = load_data("data/mvtec_ad/screw", batch_size=16)

    # 初始化模型和优化器
    model = StudentTeacherModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for images in train_loader:
            images = images.cuda()
            teacher_out, student_out = model(images)

            # 计算损失
            loss = torch.nn.functional.mse_loss(student_out, teacher_out)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 记录每个epoch的损失
        logger.info(f"Epoch [{epoch + 1}/10], Loss: {loss.item()}")

if __name__ == "__main__":
    train()
