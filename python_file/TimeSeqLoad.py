#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch

bikes_numpy = np.loadtxt('../data/p1ch4/bike-sharing-dataset/hour-fixed.csv',
                        dtype=np.float32,
                        delimiter=',',
                        skiprows=1,
                        converters={1: lambda x: float(x[8:10])}
                        )
bikes = torch.from_numpy(bikes_numpy)
bikes


# In[5]:


bikes.shape


# In[6]:


daily_bikes = bikes.view(-1, 24, bikes.shape[1]) #重塑張量

daily_bikes.shape


# In[7]:


daily_bikes = daily_bikes.transpose(1, 2)
daily_bikes.shape
# torch.Size([730, 17, 24]) => B x C x L(B: 表示有多少天的資料， C: 為有多少種資訊， L: 一天中有多少以資料)


# In[8]:


first_day = bikes[:24].long() #取出第一天(首 2個小時)的資料
weather_onehot = torch.zeros(24, 4) #先創建一個 shape 為 24x4 的張量，內部直接為初始化為 0，用於存放 one-hot編碼後的結果

first_day[:,9] #取出第一天的所有資料中，第 9 行的資訊(即天氣狀況)


# In[12]:


weather_onehot.scatter_(1, #沿著第 1 軸進行 one_hot 編碼
                       first_day[:,9].unsqueeze(1).long() - 1, #進行 one-hot 編碼的來源張量，因為原始資料是 1~4 所以要 -1
                       1.0) #設定 one-hot 編碼中非 0 的值為 1  


# In[19]:


torch.cat((bikes[:24], weather_onehot), 1)[:1] #將原始資料與 one-hot 天氣情況沿著 1軸的方向接起來


# In[21]:


#將 dauky_baikes 做上述的 天氣情況 one-hot處理
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
daily_weather_onehot.shape


# In[22]:


daily_weather_onehot.scatter_(1, daily_bikes[:,9,:].long().unsqueeze(1) - 1, 1.0)
daily_weather_onehot.shape


# In[23]:


daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)


# In[26]:


#也可以把天氣狀況當作連續數值，並將它投影到 0.0 ~ 1.0之間
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0 #在把天氣狀況行的值都減了 1 後，最大值為 3，若將該行的值除以 3，就可以將天氣狀況值限制在 0~1 之間


# In[27]:


#另一種將他們的值域映射到[0.0, 1.0]的選擇如下
temp = daily_bikes[:, 10, :] #這裡以資料集中的 攝氏溫度 資訊為例，其所引為 10
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min)


# In[28]:


#也可以做標準化
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - torch.mean(temp)) / torch.std(temp)

