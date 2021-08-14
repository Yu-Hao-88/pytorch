#!/usr/bin/env python
# coding: utf-8

# In[5]:


import imageio

dir_path = "../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM') #讀取檔案，存放在 vol_arr
vol_arr.shape #輸出 vol_arr 的 shape
#(99, 512, 512) => (張數, (尺寸)) 通道軸被省略掉了


# In[9]:


import torch

vol = torch.from_numpy(vol_arr).float() #先把資料轉成浮點數張量
vol = torch.unsqueeze(vol, 1) #在第 1 軸的位置增加插入一個通道軸 維度為 1
vol = torch.unsqueeze(vol, 2) #在第 2 軸的位置增加插入一個深度軸 維度為 1

vol.shape # N x C x D x H x W   D = depth

