import torch
from models.student_teacher import StudentTeacherModel
from PIL import Image
from torchvision import transforms
import os

def inference(image_path, gt_path=None):
    """
    用于推理单张图像并计算异常分数
    Args:
        image_path (str): 输入图像路径
        gt_path (str, optional): 如果有标注（ground truth），提供该路径
    """
    # 加载模型
    model = StudentTeacherModel().cuda()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 加载输入图像
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).cuda()  # 增加 batch 维度

    # 加载对应的标注图像（如果提供了标注路径）
    ground_truth = None
    if gt_path:
        ground_truth = Image.open(gt_path).convert("1")  # 转为二值图
        ground_truth = transform(ground_truth).unsqueeze(0).cuda()  # 增加 batch 维度

    with torch.no_grad():
        teacher_out, student_out = model(image)

        # 计算异常分数：使用 MSE 损失作为异常度量
        score = torch.nn.functional.mse_loss(student_out, teacher_out).item()

        print(f"Anomaly score for {image_path}: {score}")

        if ground_truth is not None:
            # 在有标注的情况下，计算模型与标注的匹配程度（比如计算精度、召回率等）
            print(f"Ground truth for {image_path} loaded. Use this to evaluate model performance.")

    return score

if __name__ == "__main__":
    image_path = "path/to/image.png"  # 替换为实际图像路径
    gt_path = "path/to/ground_truth_mask.png"  # 如果有标注文件，替换为实际路径
    inference(image_path, gt_path)  # 调用推理函数
