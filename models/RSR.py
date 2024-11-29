import torch
import torch.nn as nn
from torch.fft import fft2, ifft2
import torch.nn.functional as F


class DifferentiableFrequencyFilter(nn.Module):
    def __init__(self, input_channels=3, embed_dim=64):
        """
        初始化频域滤波模块
        - input_channels: 输入图像的通道数
        - embed_dim: 嵌入维度，用于生成滤波器
        """
        super(DifferentiableFrequencyFilter, self).__init__()

        # 用于生成滤波器的编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 1, kernel_size=1)  # 生成单通道滤波器
        )

        # 用于模拟 top-k 效果的可微函数
        self.sparsity_weight = nn.Parameter(torch.tensor(10.0))  # 控制稀疏程度的参数

    def forward(self, img_tensor):
        """
        前向传播：完整的频域滤波流程
        - img_tensor: 输入图像 (B, C, H, W)
        返回：处理后的图像张量 (B, C, H, W)
        """
        # # 如果输入是多通道图像，先转为灰度图 (B, 1, H, W)
        # if img_tensor.shape[1] > 1:
        #     img_tensor = img_tensor.mean(dim=1, keepdim=True)

        # 1. 傅里叶变换
        dft_img = fft2(img_tensor)
        magnitude = torch.abs(dft_img)  # 幅度
        phase = torch.angle(dft_img)  # 相位

        # 2. 滤波器生成 (输入幅度图)
        embedding = self.encoder(magnitude)

        # 3. 使用 Softmax 模拟稀疏化
        sparse_weights = F.softmax(embedding * self.sparsity_weight, dim=-1)  # 在空间维度归一化

        # 4. 应用滤波器到幅度图
        filtered_magnitude = magnitude * sparse_weights.squeeze(1)  # 滤波器与幅度相乘

        # 5. 结合相位，重建频域图
        filtered_dft = filtered_magnitude * torch.exp(1j * phase)

        # 6. 逆傅里叶变换
        filtered_img = ifft2(filtered_dft).real

        return filtered_img


# 示例使用
def main():
    import cv2
    # 假设输入图像是一个batch的图像 (B, C, H, W)
    img_path = '/home/suguilin/MMFS/datasets/MFNet/RGB/00064D.png'
    img = cv2.imread(img_path)  #torch.randn(1, 3, 256, 256)  # 示例图像 (1, 3, 256, 256)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    # img = torch.randn(1, 1, 256, 256)  # 灰度图像

    # 创建频域模块实例
    model = DifferentiableFrequencyFilter(input_channels=3, embed_dim=64)

    # 前向传播
    filtered_img = model(img)
    save_img = filtered_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    cv2.imwrite("freq_transform.png",save_img)

    print(f"输出图像形状: {filtered_img.shape}")

if __name__ == '__main__':
    main()
