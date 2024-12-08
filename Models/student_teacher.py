import torch
import torch.nn as nn

class StudentTeacherModel(nn.Module):
    def __init__(self):
        super(StudentTeacherModel, self).__init__()
        # 示例：简单的 CNN 模型
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 256 * 256, 1024)
        self.fc2 = nn.Linear(1024, 1)  # 输出一个值，表示图像的异常分数

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        # 返回教师输出和学生输出
        return out, out  # 在这里我们使用同一个输出作为教师和学生的输出，实际上可以根据需要进行修改
