import torch
import torch.nn as nn

class PatchDescriptionNetwork(nn.Module):
    def __init__(self):
        super(PatchDescriptionNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # 测试网络
    model = PatchDescriptionNetwork()
    sample_input = torch.randn(1, 3, 256, 256)  # 假设输入为 256x256 图片
    output = model(sample_input)
    print(f"Output shape: {output.shape}")  # 应输出 [1, 384, 15, 15]

