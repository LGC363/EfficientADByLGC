import torch
from models.student_teacher import StudentTeacherModel
from PIL import Image
from torchvision import transforms

def inference(image_path):
    # 加载模型
    model = StudentTeacherModel().cuda()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        teacher_out, student_out = model(image)
        # 输出预测的异常分数（例如使用 MSE）
        score = torch.nn.functional.mse_loss(student_out, teacher_out).item()
        print(f"Anomaly score: {score}")

if __name__ == "__main__":
    image_path = "path/to/image.png"  # 替换为实际图像路径
    inference(image_path)

