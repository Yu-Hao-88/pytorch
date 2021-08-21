#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import datasets

data_path = '../data/p1ch7' #下載存放路徑
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)#產生訓練資料集物件(若資料未曾被下載至本機，TorchVision 便會進行下載)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True) #當指定 Train=False 會下載驗證資料集


# In[2]:


type(cifar10).__mro__  #先用 type() 取得所屬類別，再用 __mro__取得類別的繼承順序


# In[3]:


len(cifar10)


# In[4]:


#建立一個 class_names 的字典
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img, label = cifar10[99] #存取 cifar10 資料集中索引為 99 的項目
img, label, class_names[label]
#RGB 圖
#圖片大小 32x32
#類別索引 1
#索引對應的類別 automobile


# In[5]:


from matplotlib import pyplot as plt

plt.imshow(img)
plt.show()


# In[6]:


from torchvision import transforms
dir(transforms)


# In[7]:


to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t.shape #印出轉換出來的張量 shape C x H x W


# In[25]:


#將 ToTensor() 當成輸入參數
tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())


# In[11]:


img_t, _ = tensor_cifar10[99] #將 tensor_cifar10 中索引為 99 的項目存入 img_t
type(img_t) #輸出 img_t 的資料類型


# In[15]:


img_t.shape, img_t.dtype


# In[16]:


img_t.min(), img_t.max() #輸出 img_t 的最小值及最大值
#原始 PIL 圖片中，各像素質的範圍在 0~255 之間，經過 ToTensor 轉換後這些值會變成 32位元浮點數，範圍在 0.0~1.0 之間


# In[17]:


#確認張量代表的圖片和之前相同
plt.imshow(img_t.permute(1, 2, 0)) #將張量的 shape 從 C x H x W 改為 H x W x C，以符合 Matplotlib 的要求
plt.show()


# In[18]:


import torch

#將所有圖片堆疊至一個額外的軸上，方便計算平均值和標準差
imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3) #堆疊至第 3 軸
imgs.shape


# In[48]:


#計算每個通道的平均值
mean = imgs.view(3, -1).mean(dim=1)
mean
#view(3, -1)會指定保留第 0 軸(通道軸)，其餘軸的維度則合併在第 1 軸
#因此 原本為 3x32x32x50000 -> 3x51200000
#接著就可以沿著第 1 軸計算各個通道中元素的平均值


# In[47]:


#計算每個通道的標準差
std = imgs.view(3, -1).std(dim=1)
std


# In[113]:


transformed_cifar10 = datasets.CIFAR10( #對訓練資料集進行正規化
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(mean.tolist()), #個通道的平均值 (0.4914, 0.4822, 0.4465)
                             tuple(std.tolist())) #個通道的標準差 (0.2470, 0.2435, 0.2616)
    ]))

cifar10_val = datasets.CIFAR10( #對驗證資料集進行正規化
    data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(mean.tolist()), #驗證集必須使用和訓練集相同的平均值及標準差來正規化
                             tuple(std.tolist()))
    ]))
cifar10_val


# In[59]:


img_t, _ = transformed_cifar10[99]
plt.imshow(img_t.permute(1, 2, 0))
plt.show()

#某些 RGB 值會因為正規化而超出 0.0~1.0 的範圍，進而改變整個通道的數值量級，Matplotlib 會北超出範圍的像素以黑色表示


# In[115]:


#7.2 區分鳥和飛機
label_map = {0: 0, 2: 1} #將飛機的類別標籤對應到 0 (原本為 0)；將小鳥的標籤對應到 1 (原本為 2)
class_name = ['airplane', 'bird']
cifar2 = [(img, label_map[label])  #取出原有訓練資料集中，飛機及小鳥的圖片
           for img, label in transformed_cifar10
           if label in [0, 2]]
cifar2_val = [(img, label_map[label]) #取出原有驗證資料集中，飛機及小鳥的圖片
               for img, label in cifar10_val
               if label in [0, 2]]


# In[61]:


import torch.nn as nn

n_out = 2
#一張圖片的 shape 為 32x32x5 = 3072
model = nn.Sequential( 
    nn.Linear(3072, 512), #(輸入特徵數, 隱藏層大小(輸出特徵數))
    nn.Tanh(),
    nn.Linear(512, n_out) #(上一層隱藏層的大小(輸入特徵數), 輸出特徵數(類別數量))
)


# In[62]:


#softmax 介紹
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum() #使用 exp() 可將輸入元素指數化


# In[63]:


x = torch.tensor([1.0, 2.0, 3.0])
softmax(x)


# In[64]:


softmax(x).sum()


# In[66]:


softmax = nn.Softmax(dim=1) #將 softmax 函數套用在第 1 軸(沿著同一列不同行)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [1.0, 2.0, 3.0]])
softmax(x)


# In[68]:


model = nn.Sequential( 
    nn.Linear(3072, 512), #(輸入特徵數, 隱藏層大小(輸出特徵數))
    nn.Tanh(),
    nn.Linear(512, n_out), #(上一層隱藏層的大小(輸入特徵數), 輸出特徵數(類別數量))
    nn.Softmax(dim=1)
)


# In[88]:


#取出一個鳥類的圖片
img, label = cifar2[0] #取出 cifar2 中索引為 0 的圖片
plt.imshow(img.permute(1, 2, 0))
plt.show()


# In[75]:


img_batch = img.view(-1).unsqueeze(0) #轉為 1D 張量，並在第 0 軸加入一個軸(批次軸)


# In[78]:


out = model(img_batch)
out
#(飛機的機率, 小鳥的機率)


# In[80]:


_, index = torch.max(out, dim=1)
index #代筆索引為 0 的類別有較高的機率值


# In[81]:


#當機率值趨近於 0 時，機率的對數運算並不容易
#故用 nn.LogSoftmax 取代 nn.Softmax
model = nn.Sequential(nn.Linear(3072, 512),
                      nn.Tanh(),
                      nn.Linear(512, 2),
                      nn.LogSoftmax(dim=1))


# In[83]:


#使用 NLL 當作 loss function 的原因可以看 readme '分類任務的損失'
loss = nn.NLLLoss()


# In[92]:


out = model(img.view(-1).unsqueeze(0))
loss(out, torch.tensor([label])) #(模型的輸出(模型預測的類別), 標籤張量(實際的類別))
#該張圖片的 loss


# In[100]:


import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(3072, 512),
                      nn.Tanh(),
                      nn.Linear(512, 2),
                      nn.LogSoftmax(dim=1))
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 100
model.cuda()
for epoch in range(1, n_epochs + 1):
    for img, label in cifar2:
        #img = to_tensor(img)
        out = model(img.view(-1).unsqueeze(0).cuda())
        loss = loss_fn(out, torch.tensor([label]).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss))) #每個迴圈結束後，將損失列印出來


# In[101]:


train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True) #(Dataset 物件, batch_size, 是否shuffle)


# In[103]:


import torch
import torch.nn as nn
model = nn.Sequential(nn.Linear(3072, 512),
                      nn.Tanh(),
                      nn.Linear(512, 2),
                      nn.LogSoftmax(dim=1))
learning_rate = 1e-2

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 100
for epoch in range(1, n_epochs + 1):
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        ouputs = model(imgs.view(batch_size, -1))
        loss = loss_fn(ouputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss))) #列印隨機批次的損失值


# In[116]:


val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy: %f" %(correct/total))


# In[117]:


model = nn.Sequential(nn.Linear(3072, 1024),
                      nn.Tanh(),
                      nn.Linear(1024, 512),
                      nn.Tanh(),
                      nn.Linear(512, 128),
                      nn.Tanh(),
                      nn.Linear(128, 2),
                      nn.LogSoftmax(dim=1))


# In[120]:


model = nn.Sequential(nn.Linear(3072, 1024),
                      nn.Tanh(),
                      nn.Linear(1024, 512),
                      nn.Tanh(),
                      nn.Linear(512, 128),
                      nn.Tanh(),
                      nn.Linear(128, 2),
                      nn.LogSoftmax(dim=1))
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[134]:


def training(n_epochs, model, loss_fn, optimizer, training_dataset, batch_size):
    model.train()
    model.cuda()
    for epoch in range(1, n_epochs + 1):
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        for imgs, labels in train_loader:
            outputs = model(imgs.view(imgs.shape[0], -1).cuda())
            loss = loss_fn(outputs, labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: %d, Loss: %f" %(epoch, float(loss)))


# In[140]:


def valid(model, val_dataset, batch_size):
    model.eval()
    model.cuda()
    val_dataload = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for imgs, labels in val_dataload:
        outputs = model(imgs.view(imgs.shape[0], -1).cuda())
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += ((labels.cuda()==predicted).sum())
    print("Accuracy: %f" %(correct/total))


# In[135]:


training(
    n_epochs = 100,
    model = model, 
    loss_fn = loss_fn, 
    optimizer=optimizer, 
    training_dataset = cifar2, 
    batch_size = 64
)


# In[141]:


valid(
    model = model, 
    val_dataset = cifar2_val, 
    batch_size = 64
)


# In[145]:


#計算模型可訓練參數
numel_list = [
    p.numel()
    for p in model.parameters()
    if p.requires_grad == True #只計算可訓練參數的數量
]
sum(numel_list), numel_list

