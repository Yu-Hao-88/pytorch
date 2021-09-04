import math
import random
from collections import namedtuple

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.logconf import logging
from util.unet import UNet

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):  # kwargs 為 python 字典，其中包含所有欲傳進 U-Net 建構子中的關鍵參數
        super().__init__()

        # 使用 BatchNorm2d 時須設定輸入的通道數，通過 kwargs 的 in_channels 關鍵字參數來指定
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()  # 客製化的模型權重初始化函式

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

        # nn.init.constant_(self.unet.last.bias, -4)
        # nn.init.constant_(self.unet.last.bias, 4)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output


class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(
            input_g.shape[0], -1, -1)  # 注意，我們擴增的是 2D 資料
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:, :2],  # 轉換結果的第 0 軸代表批次，對於批次中每一個元素而言，我們只取 3x3 矩陣前兩列
                                 input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g,
                                          affine_t, padding_mode='border',
                                          align_corners=False)
        # 由於需對 CT 和遮罩進行相同的轉換，故這裡使用一樣的 grid。注意 grid_sample 只能處理浮點數，故須將資料改成適當型別
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                                          affine_t, padding_mode='border',
                                          align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        # 在傳回結果以前，透過與 0.5 的比較 將 augmented_label 變成布林張量
        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)  # 建立一個 3x3 的對角矩陣，但稍後會捨棄最後一列

        for i in range(2):  # 再次強調，此處擴增的是 2D 資料
            if self.flip:
                if random.random() > 0.5:  # 有 50% 的機率會翻轉
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            # 此處產生一個隨機單位的弧度(radians)的角度，該角度範圍在 0 到 2pi 之間
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([  # 此旋轉矩陣能對前兩軸進行隨機角度的 2D 旋轉
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t  # 使用 Python 的矩陣乘法算符來旋轉轉換矩陣

        return transform_t


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        # 尾部
        self.tail_batchnorm = nn.BatchNorm3d(1)  # 批次正規化模組

        # 主體
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        # 頭部
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {  # 處理內有可訓練參數的模型
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(  # 一種特殊的正規化方法
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)  # 將偏值限制在一定的範圍內

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)  # 尾部的輸出

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)  # 主體的輸出

        conv_flat = block_out.view(  # 將輸出扁平化
            block_out.size(0),  # 一批次的大小
            -1,
        )
        linear_output = self.head_linear(conv_flat)  # 頭部線性層可接受扁平化的輸出

        # 原始的模型輸出, 經過 softmax 處理後的模型輸出(機率值)
        return linear_output, self.head_softmax(linear_output)


class LunaBlock(nn.Module):  # 在建立區塊時要指定輸入集輸出的 channel 數
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(  # 第一個卷積模組
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(  # 第二個卷積模組
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)  # 最大池化層

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
