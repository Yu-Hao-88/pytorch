#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv

wine_path = '../data/p1ch4/tabular-wine/winequality-white.csv' #設定資料路徑
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1) #delimiter 分隔符號，skiprows 跳過的列數

wineq_numpy


# In[2]:


col_list = next(csv.reader(open(wine_path), delimiter=';')) #取得標題列所有特徵名稱

wineq_numpy.shape, col_list #輸出 wineq_numpy 陣列的 shape 及所有特徵名稱


# In[3]:


import torch

wineq = torch.from_numpy(wineq_numpy)
wineq.shape, wineq.dtype


# In[4]:


data = wineq[:, :-1] #選擇除了最後一行(品質分數)以外的所有資料
data, data.shape


# In[5]:


target = wineq[:, -1] #取出 wineq 中的品質分數，做為目標張量
target, target.shape


# In[6]:


target = wineq[:, -1].long() #將 wineq 中的品質分數轉為整數
target


# In[7]:


target_onehot = torch.zeros(target.shape[0], 10) #建立 shape 為 [4980, 10]，值均為 0 的張量，用來存放編碼結果

a = target_onehot.scatter_(1, target.unsqueeze(1), 1.0) #依 targe 的內容進行編碼

#scatter_ 有 3 個參數
# 1. 要沿著哪個軸進行 one-hot 編碼( 1代表第 1 軸)
# 2. 要進行 one-hot 編碼的來源張量(張量內的值表示 target_onehot 中哪一個索引位置要被設為非 0 的值)
# 3. 設定 one-hot 編碼中非 0 的值為多少，通常是 1
# 總結: 對於每一列，取目標標籤(就是第 2 個參數所指定的張量，在以上例子中，即個別樣本的品質分數)的值作為 one-hot 向量中的索引，
# 設置相應值為 1.0
# !! 來源張量(第二個參數)的軸數，必須與我們散佈(scatter)到的結果張量(target_onehot)相同，
# 由於 target_onehot有兩個軸(4898x10)因此要用 unsqueeze() 將 target(1軸向量)增加 1軸
a, a.shape


# In[8]:


target_unsqueezed = target.unsqueeze(1) #添加一個軸至第 1 軸
print(target_unsqueezed)
print(target_unsqueezed.shape)


# In[10]:


data_mean = torch.mean(data, dim=0) #計算平均值，dim=0表示指定要沿著第 0 軸做計算

data_mean #個化學性質的平均值


# In[11]:


data_var = torch.var(data, dim=0) #計算標準差
data_var


# In[12]:


data_normalized = (data - data_mean) / torch.sqrt(data_var) #用 Python 實現標準化的公式 
data_normalized


# In[13]:


bad_indexes = target <= 3
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()


# In[14]:


bad_data = data[bad_indexes] #利用索引篩選出符合目標的項目
bad_data.shape


# In[18]:


bad_data = data[target <= 3] #篩選出品質 差 的項目

#代表邏輯運算中的 and，即要前後條件都符合，才會傳回 true
mid_data = data[(target > 3) & (target < 7)] #篩選出品質 中等 的項目
good_data = data[target >= 7] #篩選出品質 好 的項目

bad_mean = torch.mean(bad_data, dim=0) #計算品質 差 的項目中，各化學性質的平均值
mid_mean = torch.mean(mid_data, dim=0) #計算品質 中等 的項目中，各化學性質的平均值
good_mean = torch.mean(good_data, dim=0) #計算品質 好 的項目中，各化學性質的平均值

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args)) #印出結果


# In[21]:


total_sulfur_threshold = 141.83 #設定閾值為 141.83，若超出此數字則為品質差的酒，低於此數字則為好酒
total_sulfur_data = data[:, 6] #取出資料集的總二氧化硫量資訊

predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold) #lt 可以挑出 A 中，值 < B 的項目索引

predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum() #輸出小於閾值的總樣本數(即預測的好酒數量)


# In[22]:


actual_indexes = target > 5
actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()


# In[23]:


#actual_indexes 實際為好酒
#predicted_indexes 預測為好酒
#利用 & 找出皆為 True 的項目索引
n_matches = torch.sum(actual_indexes & predicted_indexes).item() #計算結果吻合的項目數量

n_predicted = torch.sum(predicted_indexes).item() #預測為好酒的項目數量
n_actual = torch.sum(actual_indexes).item() #實際為好酒的項目數量
n_matches, n_matches/n_predicted, n_matches/n_actual

