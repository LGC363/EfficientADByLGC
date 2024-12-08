import torch
from models.student_teacher import StudentTeacherModel
from utils.dataset import load_data
from sklearn.metrics import roc_auc_score

def evaluate():
    # 加载数据
    test_loader = load_data("data/mvtec_ad/screw", batch_size=16)
    model = StudentTeacherModel().cuda()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images in test_loader:
            images = images.cuda()
            teacher_out, student_out = model(images)
            # 获取预测结果和标签
            # 假设使用 MSE 损失，作为异常检测指标
            preds = torch.nn.functional.mse_loss(student_out, teacher_out, reduction='none').mean(dim=(1, 2, 3))
            labels = torch.zeros_like(preds)  # 使用真实标签进行标记（正常=0, 异常=1）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"AU-ROC Score: {auc_score}")

if __name__ == "__main__":
    evaluate()

