import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MVTECDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): 数据集的根目录，包含 train、test 和 ground_truth 子目录。
            mode (str): 模式，可以是 'train' 或 'test'，默认是 'train'。
            transform (callable, optional): 可选的转换操作（如数据增强、归一化等）。
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # 设置图像路径和对应的 ground truth 路径
        self.image_paths, self.ground_truth_paths = self._load_image_paths()

    def _load_image_paths(self):
        """
        根据 mode ('train' 或 'test') 来加载相应的图像路径
        """
        image_paths = []
        ground_truth_paths = []

        # 获取对应模式下的文件路径（train 或 test）
        mode_dir = os.path.join(self.root_dir, self.mode)
        if not os.path.isdir(mode_dir):
            raise ValueError(f"Mode '{self.mode}' not found in {self.root_dir}")

        if self.mode == 'train':
            # 训练集只包含正常图像
            good_dir = os.path.join(mode_dir, 'good')
            if os.path.exists(good_dir):
                for file_name in os.listdir(good_dir):
                    if file_name.endswith('.png'):
                        image_paths.append(os.path.join(good_dir, file_name))

        elif self.mode == 'test':
            # 测试集包含正常和异常图像
            for category in ['good', 'manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top']:
                category_dir = os.path.join(mode_dir, category)
                if os.path.exists(category_dir):
                    for file_name in os.listdir(category_dir):
                        if file_name.endswith('.png'):
                            image_paths.append(os.path.join(category_dir, file_name))

                    # Ground truth 掩码是与测试图像一一对应的，命名格式为 000_mask.png
                    gt_dir = os.path.join(self.root_dir, 'ground_truth', category)
                    if os.path.exists(gt_dir):
                        for file_name in os.listdir(gt_dir):
                            if file_name.endswith('.png') and file_name.replace('_mask', '') in os.listdir(category_dir):
                                ground_truth_paths.append(os.path.join(gt_dir, file_name))

        return image_paths, ground_truth_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 根据给定索引获取图像路径
        img_path = self.image_paths[idx]
        # 加载图像并转换为 RGB
        image = Image.open(img_path).convert("RGB")

        # 加载标注图像（如果是测试集）
        ground_truth = None
        if self.mode == 'test':
            # 对应的 ground_truth 路径，文件名格式为 000_mask.png
            gt_path = self.ground_truth_paths[idx]
            ground_truth = Image.open(gt_path).convert("1")  # 1 表示二值图（标注图像）

        # 如果提供了数据转换（如数据增强、归一化等），则应用
        if self.transform:
            image = self.transform(image)
            if ground_truth:
                ground_truth = self.transform(ground_truth)

        # 返回图像和标注（训练集只有图像，测试集返回图像和标注）
        if self.mode == 'test':
            return image, ground_truth
        else:
            return image
