import torch
from models.student_teacher import StudentTeacherModel
from utils.dataset import MVTECDataset  # 使用修改后的 MVTECDataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms

def evaluate():
    # 数据转换：将图像大小调整为 256x256
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像调整为 256x256
        transforms.ToTensor(),
    ])

    # 创建测试数据集并加载
    test_dataset = MVTECDataset(root_dir="data/mvtec_ad/screw", mode='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 加载模型
    model = StudentTeacherModel().cuda()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, ground_truth in test_loader:  # 获取测试数据和标注
            images = images.cuda()
            ground_truth = ground_truth.cuda()

            teacher_out, student_out = model(images)

            # 计算 MSE 损失，作为异常检测指标
            preds = torch.nn.functional.mse_loss(student_out, teacher_out, reduction='none').mean(dim=(1, 2, 3))

            # 这里假设 ground_truth 已经是标记正常（0）或异常（1）的标签
            labels = ground_truth  # 这是一个简化的假设，实际使用时，ground_truth 可能是二值图像

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算 AU-ROC
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"AU-ROC Score: {auc_score}")

    # 可选：计算 Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds > 0.5, average='binary')
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

if __name__ == "__main__":
    evaluate()
