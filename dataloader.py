import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        """
        初始化数据集.
        Args:
            low_dir (str): 低光照图像文件夹路径.
            high_dir (str): 正常光照图像文件夹路径.
            transform (callable, optional): 应用于图像的转换.
            crop_size (int, optional): 训练时随机裁剪的尺寸.
            training (bool): 是否为训练模式.
        """
        super().__init__()
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        # 获取文件夹下所有文件的文件名列表，并排序以确保低光和高光图像一一对应
        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

        # 断言检查，确保两个文件夹中的图片数量一致
        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        """返回数据集中图像的总数."""
        return len(self.low_images)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个数据样本.
        返回一个元组: (low_light_image, high_light_image, filename)
        """
        # --- 核心修改在这里 ---
        # self.low_images[idx] 本身就是我们想要的图片文件名 (例如: "001.png")
        image_filename = self.low_images[idx]

        # 使用文件名构建完整的图像路径
        low_image_path = os.path.join(self.low_dir, image_filename)
        high_image_path = os.path.join(self.high_dir, self.high_images[idx]) # 假设高光图片名与低光一致

        # 打开图像
        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')

        # 应用预设的 transform (例如 ToTensor, Normalize)
        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        # 如果是训练模式并且设置了裁剪尺寸，则进行随机裁剪
        if self.training and self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(low_image, output_size=(self.crop_size, self.crop_size))
            low_image = transforms.functional.crop(low_image, i, j, h, w)
            high_image = transforms.functional.crop(high_image, i, j, h, w)

        # 返回三个值：低光图像、高光图像、文件名
        return low_image, high_image, image_filename

def create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = None
    test_loader = None
    
    if train_low and train_high:
        train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    if test_low and test_high:
        test_dataset = PairedDataset(test_low, test_high, transform=transform, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader

if __name__ == '__main__' :  
    train_low = 'TEDLOLv1/Train/input'
    train_high = 'TEDLOLv1/Train/target'
    test_low = 'TEDLOLv1/Test/input'
    test_high = 'TEDLOLv1/Test/target'

    train_loader, test_loader, image_names= create_dataloaders(train_low=train_low,
                                                    train_high=train_high,
                                                    test_low=test_low,
                                                    test_high=test_high,
                                                    crop_size=256,
                                                    batch_size=10)
    i = 0 
    #for batch_idx, batch in enumerate(test_loader):
    for batch_idx, batch in enumerate(train_loader):
            low,high = batch[0],batch[1]
            i +=1
            if i < 10:
                print('low.shape={}, high.shape={}'.format(low.shape,high.shape))
        # low, high = enumerate(test_loader)

