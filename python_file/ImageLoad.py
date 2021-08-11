#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
import imageio

img_arr = imageio.imread('../data/p1ch4/image-dog/bobby.jpg')
img_arr.shape #輸出 img_arr 的 shape
#(720, 1280, 3) => ((空間大小), 色彩通道數) img_arr 為 1 個 3 軸張量


# In[4]:


img = torch.from_numpy(img_arr) #先將 img_arr 轉成 PyTorch 張量(以 img 表示) 
out = img.permute(2, 0, 1) #將第 2 軸移至第 0 軸，原本的第 0 軸與第 1 軸往後移
out.shape #輸出調整後的軸順序


# In[6]:


batch_size = 100 #設定一次可載入 100 張圖片
batch = torch.zeros(batch_size, 3, 256, 256, dtype = torch.uint8) #此張量可儲存 100 張高度與寬度都是 256 像素的 RBG 圖片


# In[29]:


import os
data_dir = '../data/p1ch4/image-cats/' #設定圖檔所在的資料夾路徑
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png'] #找出所有 png

for i, filename in enumerate(filenames): #依序載入每個 png 檔
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1) #調整張量中各軸的排列順序
    img_t = img_t[:3] #有些圖片會友表示透明度的第 3 軸，是我們不需要的，故只保留前面 3 軸(H, W, C)
    
    batch[i] = img_t
batch.shape
#torch.Size([100, 3, 256, 256]) => (N X C X H X W)


# In[30]:


#對像素值進行正規化，常見的做法是將像素質除以 255
batch = batch.float() #將 batch 張量內的像素值轉換成浮點數
batch /= 255.0 #將像素值同除以 255


# In[32]:


#另一種方法是做標準化
n_channels = batch.shape[1] #取得色彩通道數(batch張量的 shape 中，第 1 軸維度)

for c in range(n_channels): #依次走訪每個色彩通道
    mean = torch.mean(batch[:, c]) #計算平均值
    std = torch.std(batch[:, c]) #計算標準差
    batch[:, c] = (batch[:, c] - mean) /std #正規化公式

