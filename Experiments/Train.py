import torch
import torch.optim as optim
from models.student_teacher import StudentTeacherModel
from utils.dataset import MVTECDataset  # 使用修改后的 MVTECDataset
from torch.utils.data import DataLoader
from utils.logger import setup_logger
from torchvision import transforms

# 设置日志
logger = setup_logger()

def train():
    # 数据转换：将图像大小调整为 256x256
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像调整为 256x256
        transforms.ToTensor(),
    ])

    # 加载训练数据集
    train_dataset = MVTECDataset(root_dir="data/mvtec_ad/screw", mode='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 初始化模型和优化器
    model = StudentTeacherModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练过程中的最优损失
    best_loss = float('inf')

    for epoch in range(10):
        model.train()
        epoch_loss = 0.0
        for images in train_loader:
            images = images.cuda()
            teacher_out, student_out = model(images)

            # 计算损失
            loss = torch.nn.functional.mse_loss(student_out, teacher_out)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 记录每个epoch的损失
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/10], Loss: {avg_loss}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')  # 保存最佳模型
            logger.info(f"Best model saved with loss: {best_loss}")

if __name__ == "__main__":
    train()
