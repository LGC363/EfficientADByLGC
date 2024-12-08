import torch
import torch.nn as nn
from models.feature_extractor import PatchDescriptionNetwork 

class StudentTeacherModel(nn.Module):
    def __init__(self):
        super(StudentTeacherModel, self).__init__()
        self.teacher = PatchDescriptionNetwork()
        self.student = PatchDescriptionNetwork()

    def forward(self, x):
        teacher_features = self.teacher(x)
        student_features = self.student(x)
        return teacher_features, student_features

if __name__ == "__main__":
    model = StudentTeacherModel()
    sample_input = torch.randn(1, 3, 256, 256)
    teacher_out, student_out = model(sample_input)
    print(f"Teacher Output: {teacher_out.shape}, Student Output: {student_out.shape}")
