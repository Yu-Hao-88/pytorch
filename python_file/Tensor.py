#!/usr/bin/env python
# coding: utf-8

# In[1]:


#3.2 多維張量
a = [1.0, 2.0, 1.0]


# In[2]:


a[0]


# In[3]:


a[0] = 3.0
a


# In[4]:


import torch #匯入 torch 模組
a = torch.ones(3) #創建一個 3 維的 1 軸張量，其元素皆為 1 
a


# In[5]:


a[1] #印出索引為 1 的元素


# In[6]:


float(a[1]) #將 a[1] 轉換為浮點數後再輸出


# In[7]:


a[2] = 2.0 #修改索引為 2 的元素
a


# In[8]:


points = torch.zeros(6) #利用 zeros() 來創建初始元素值皆為 0 的 1 軸張量
points[0] = 4.0 #點 A 的 x 座標
points[1] = 1.0 #點 A 的 y 座標
points[2] = 5.0 #點 B 的 x 座標
points[3] = 3.0 #點 B 的 y 座標
points[4] = 2.0 #點 C 的 x 座標
points[5] = 1.0 #點 C 的 y 座標
points


# In[9]:


points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0]) #存有座標值的串列
points


# In[10]:


#點 A 的 x 座標, 點 A 的 x 座標
float(points[0]), float(points[1])


# In[11]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points[0] #用索引 0 來提取點 A 的座標值


# In[12]:


points.shape #shape 會回傳張量在每個軸的維度
# 輸出: torch.Size([3, 2]) 表示 points 為 2 軸張量，第 0 軸有 3 維，第 1 軸有 2 維。


# In[13]:


points = torch.zeros(3, 2) #創建一個元素值全為 0 的張量，並指定期 shape 為(3,2)
points


# In[14]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]]) #先創建一個 2 軸張量
points


# In[15]:


points[0, 1] #取出第 0 列，第 1 行的元素


# In[16]:


#3.3 利用索引值操作張量
some_list = list(range(6)) #產生一個 0 到 5 的數字串列
some_list[:] #提取串列中所有元素


# In[17]:


some_list[0:4] #提取第 0 個到第 3 個元素


# In[18]:


some_list[1:] #提取第 1 個元素到最後一個元素


# In[19]:


some_list[:4] #提取第 0 個元素到第 3 個元素


# In[20]:


some_list[:-1] #提取第 0 個到倒數第二個元素


# In[21]:


some_list[1:4:2] #從第 1 個元素到第 3 個元素，每次間隔 2 個元素


# In[22]:


points #利用前面的 points 張量


# In[23]:


points[1:, :] #提取第 1 列起所有元素


# In[24]:


points[1:, 0] #提取第 1 列起，位於第 0 行的元素


# In[25]:


#為張量命名

#將第 0 軸命名為channels，此張量只有一個軸，shape 為 (3,)
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
weights_named


# In[26]:


img_t = torch.randn(3, 5, 5) #創建一個 3 軸張量
batch_t = torch.randn(2, 3, 5, 5) #創建一個 4 軸張量

img_named = img_t.refine_names(..., 'channels', 'rows', 'columns') #針對張量最後三個軸分別命名，'...'表示要略過該軸
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')

print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names) #未命名的軸以 none 表示


# In[27]:


#weights_named 為 1 軸張量，只有 channels 軸
weights_aligned = weights_named.align_as(img_named) #align_as 會將前者對齊後者的軸順序並自動擴張
weights_aligned.shape, weights_aligned.names #變成 1 個 3 軸張量，自動添加 rows 和 columns 軸


# In[28]:


(img_named * weights_aligned)


# In[29]:


gray_named = (img_named * weights_aligned).sum('channels') #兩個張量相乘後，將 channels 軸內的子陣列加總
gray_named.shape, gray_named.names


# In[30]:


#若試圖將不同名稱的軸組合起來，系統會報錯:
#gray_named = (img_named[:3] * weights_named).sum('channels')


# In[31]:


gray_plain = gray_named.rename(None) #將張量變回原本未命名的樣子
gray_plain.shape, gray_plain.names


# In[32]:


#3.5 張量的元素型別
double_points = torch.ones(10, 2, dtype=torch.double) #指定該張量內的數值為 64 位元的雙精度浮點數
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short) #指定該張量內的數值為 16 位元的整數


# In[33]:


short_points.dtype #呼叫 short_point 的 dtype 屬性


# In[34]:


double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()
print(double_points.dtype)
print(short_points.dtype)


# In[35]:


#to 會檢查張量原本的數值類型與我們指定的是否一致，若不一致則轉換。
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)
print(double_points.dtype)
print(short_points.dtype)


# In[36]:


points_64 = torch.zeros(5, dtype=torch.double) #建立內含 5 個元素的 64 位元浮點數張量，其中元素值皆為 0
points_short = points_64.to(torch.short) #設定 points_short 為 16 位元的整數張量
points_64 * points_short #輸出結果為 64 位元的浮點數


# In[37]:


#3.6 其他常用的張量功能
a = torch.ones(3, 2) # 創建一個 shape 為 3x2 的 2 軸張量
a_t = torch.transpose(a, 0, 1) #將 a 張量的第 0 軸與第 1 軸轉置
a.shape, a_t.shape #印出 a 張量及 a_t 張量的 shape


# In[38]:


a = torch.ones(3, 2)
a_t = a.transpose(0, 1)#將 a 張量的第 0 軸與第 1 軸轉置，但此處是呼叫張量的 method
a.shape, a_t.shape


# In[39]:


#3.7 張量的儲存原理
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]]) #創建一個 3x2 的 2軸張量，用來存放 3個座標點
points.storage() #輸出 points 張量的 storage


# In[40]:


points_storage = points.storage() #points_storage 為 points 張量的 shape
points_storage[0] #取得 points_storage 中第 0 個元素


# In[41]:


points.storage()[0] #另一種取得 storage 中元素的方法


# In[42]:


points_storage[0] = 2.0 #將 points_storage 中索引為 0 的元素值由 4.0 改為 2.0
points


# In[43]:


a = torch.ones(3, 2) #將 a 張量內的所有元素值初始化為 1
a


# In[44]:


a.zero_() #將 a 張量內的所有元素值修改成 0 
a


# In[45]:


#3.8 大小、偏移和步長
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]]) #創建一個 3x2 的 2軸張量，用來存放 3個座標點
points.storage() #輸出 points 張量的 storage


# In[46]:


second_point = points[1] #從 points 張量中抽取索引為 1 的點，並放入 second_point 張量
second_point.storage() # 輸出 second_point 張量的 storage
#與 points 張量指向同一個 storage


# In[47]:


second_point.storage_offset() #輸出 second_point 張量的偏移量
#由於先跳過 4.0 和 1.0 因此偏移量為 2


# In[48]:


second_point.size()


# In[49]:


second_point.shape
#結果與使用 size() 相同


# In[50]:


points.stride()
#第 0 軸的索引每 +1，需要跳過 2 個元素, 第 1 軸的索引每加 1 需要跳過 1 個元素


# In[51]:


second_point = points[1] #再次產生一個 second_point 張量
second_point.size()


# In[52]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point[0] = 10.0 #將 second_point 中索引為 0 的元素值改為 10.0
points #檢查看看 points 張量的內容是否有變化


# In[53]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone() #利用 clone 複製一份 points 張量中索引為 1 的座標資料
second_point[0] = 10.0 #修改 second_point 中，索引為 0 的元素值
points #points 元素未被修改


# In[54]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

#tensor([[4., 1.], <- 代表一個座標點
#        [5., 3.],
#        [2., 1.]])
#         ^   ^ 
#         |   |
#x 座標(第0行) y座標(第1行)


# In[55]:


points_t = points.t() #對 points 張量進行轉置，並存為 points_t
points_t

#tensor([[4., 5., 2.],  <- x座標
#        [1., 3., 1.]]) <- y座標
#         ^   
#         |   
#  代表一個座標點 


# In[56]:


id(points.storage()) == id(points_t.storage()) #測試 points 張量及 points_t 張量的 storage 是否相同


# In[57]:


points.stride()


# In[58]:


points_t.stride()


# In[59]:


some_t = torch.ones(3, 4, 5) #創建一個 shape 為 3x4x5 的 3 軸張量
some_t.shape


# In[60]:


transpose_t = some_t.transpose(0, 2) #將第 0 軸及第 2 軸進行轉置
transpose_t.shape


# In[61]:


some_t.stride()


# In[62]:


transpose_t.stride()


# In[63]:


points.is_contiguous() #用 is_contiguous()來檢查某個張量是否連續


# In[64]:


points_t.is_contiguous() #轉置矩陣不是連續張量


# In[65]:


points =torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t() #對張量進行轉置
points_t


# In[66]:


points_t.storage()


# In[67]:


points_t.stride()


# In[68]:


points_t_cont = points_t.contiguous() #利用 contiguous() 將 points_t 轉換成連續的張量，並存放在 points_t_cont
points_t_cont


# In[69]:


points_t_cont.stride() #步長改變了，元素變成沿著'列'方向排列


# In[70]:


points_t_cont.storage()


# In[71]:


#3.9 把張量移到 GPU 上
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda') #將指定張量存到 GPU 上


# In[72]:


points_gpu = points.to(device='cuda') #之前的例子 points 張量是存在 CPU 上，現在將他轉移至 GPU


# In[73]:


points_gpu = points.to(device='cuda:0') #將 points 轉移到第 0 個 GPU 上


# In[74]:


points = 2 * points #在 CPU 上的乘法操作
points_GPU = 2 * points.to(device='cuda') #在 GPU 上的乘法操作
#1. points 張量的值被複製到 GPU 上
#2. 在 GPU 上建立一個新張量 (points_gpu)，用來存放接下來的乘法結果
#3. 傳回張量的計算結果


# In[75]:


points_gpu = points_gpu + 4 #一樣在 GPU 上進行


# In[76]:


points_cpu = points_gpu.to(device='cpu') #把張量搬回 cpu


# In[77]:


points_gpu = points.cuda() #將 points 張量轉移至 GPU 上
points_gpu = points.cuda(0) #將 points 張量轉移至第 0 顆 GPU 上
points_cpu = points_gpu.cpu() #將張量移回 CPU 上


# In[78]:


#3.10 與 NumPy 的互動性
points = torch.ones(3, 4) 
points_np = points.numpy() #利用 numpy() 將 PyTorch 張量轉換為 NumPy 陣列
points_np


# In[79]:


points = torch.from_numpy(points_np) #轉回 pytorch 張量


# In[80]:


#3.12 將張量序列化(長期儲存)
torch.save(points, 'ourpoints.t') #儲存張量


# In[81]:


with open('ourpoints.t','wb') as f: #儲存張量
    torch.save(points, f)


# In[82]:


points = torch.load('ourpoints.t') #讀取張量
points


# In[83]:


with open('ourpoints.t', 'rb') as f:
    points = torch.load(f)
points
#上述的方法不能以 PyTorch 以外的軟體讀取


# In[84]:


import h5py
f = h5py.File('ourpoints.hdf5', 'w') #創建一個 hdf5 檔，並開啟寫入模式
dset = f.create_dataset('coords', data=points.numpy()) #先將張量轉換為 NumPy 陣列，並作為 create_dataset()參數
f.close()


# In[91]:


f = h5py.File('ourpoints.hdf5', 'r') #讀入存有各個座標點資訊的 HDF5 檔，r 代表 read
dset =  f['coords'] #上個 cell 的 coords 為 index 因為　hdf5 為 dictionary 結構
last_points = dset[-2:] #取得最後兩個座標點 此時資料才被載入回來
last_points


# In[89]:


last_points = torch.from_numpy(dset[-2:]) #資料會被複製到張量的 storage 中
f.close()


# In[93]:


#延伸思考
a = torch.tensor(list(range(9)))
a


# In[94]:


a.size()


# In[97]:


a.storage_offset()


# In[98]:


a.stride()


# In[99]:


b = a.view(3, 3)
b


# In[100]:


id(b.storage()) == id(a.storage())


# In[102]:


c = b[1:,1:]
c


# In[103]:


c.size()


# In[104]:


c.storage_offset()


# In[105]:


c.stride()


# In[113]:


test = a.cos()
test.dtype


# In[125]:


a.to(dtype=torch.float32).cos_()

