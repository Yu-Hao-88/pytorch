#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)


# In[2]:


netG = ResNetGenerator()


# In[3]:


model_path = '../data/p1ch2/horse2zebra_0.4.0.pth' #存有模型參數的 pth 檔所在路徑

#將訓練過的參數載入 netG
model_data = torch.load(model_path) 
netG.load_state_dict(model_data)


# In[4]:


netG.eval()


# In[8]:


from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256), #定義預處理函式
    transforms.ToTensor()
])


# In[9]:


img = Image.open("../data/p1ch2/horse.jpg")
img


# In[10]:


img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0) # 在第 0 階加入一個 batch 軸，代表輸入的圖片數量


# In[16]:


batch_out = netG(batch_t)

batch_out = torch.squeeze(batch_out, 0) #將第 0 階去除，使 batch_out 變回一個 3D 張量
batch_out = (batch_out + 1.0)/2.0 #調整圖片的明暗度
out_img = transforms.ToPILImage()(batch_out) #轉換成圖片
out_img
#out_img.show()

