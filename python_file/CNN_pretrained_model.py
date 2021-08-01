#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import models
dir(models)
# 大寫開頭的名字都是依些有名的神經網路 model
# 小寫的名字則是函式，傳回前述類別的實體化(instantiated)模型，並使用不同的超參數(ex:層數)


# In[2]:


alexnet = models.AlexNet()

# 載入已經 train 好的 resnet
resnet = models.resnet101(pretrained = True)
resnet


# In[12]:


from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256), #轉換尺寸
    transforms.CenterCrop(224), #進行裁切
    transforms.ToTensor(), #轉為張量
    transforms.Normalize( #正規化處理
        mean = [0.485, 0.456, 0.406], #手動設定各色彩通道的平均值及標準差
        std = [0.229, 0.224, 0.225]
    )])


# In[10]:


from PIL import Image
img = Image.open("../data/p1ch2/bobby.jpg")
img


# In[9]:


img_t = preprocess(img)
img_t.shape


# In[14]:


'''
ResNet 規定的 input 資料須為 4D ( batch, channel, width, height )
因此使用 unsqueeze 在 index=0 增加一維
'''
import torch
batch_t = torch.unsqueeze(img_t, 0)
batch_t.shape


# In[15]:


resnet.eval()


# In[17]:


out = resnet(batch_t) #給NN輸入剛剛載入的圖片
out
out.shape #1代表 batch 量，因為目前只有一張圖片，所以為1


# In[18]:


with open('../data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


# In[32]:


_, index = torch.max(out, 1) #找出 out 張量的第一軸中，最大值的索引
index[0] #此時 index 是 1D tensor


# In[23]:


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 #[0] 應該是取用第一個 測資
labels[index[0]], percentage[index[0]].item() #分數最高的類別標籤, 最高的分數


# In[41]:


_, indices = torch.sort(out, descending=True) #對分數進行排序，從高至低
predicted_output = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]] #列出分數前5高的類別
print(predicted_output)

