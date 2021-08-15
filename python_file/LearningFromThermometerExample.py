#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

t_c = [ 0.5, 14.0, 15.0, 28.0, 11.0, 8.0,   3.0, -4.0,  6.0, 13.0, 21.0] #攝氏溫度
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] #溫度計上對應到的的讀數

t_c = torch.tensor(t_c) #將資料集打包成張量
t_u = torch.tensor(t_u)


# In[2]:


def model(t_u, w, b): #該模型的參數為: 輸入張量(t_u)、權重參數(w)與偏差參數(b)
    return w * t_u + b #傳回預測值


# In[3]:


#t_p 為預測值張量
#t_c 為實際值張量
def loss_fn(t_p, t_c): #定義損失函數 此為 MSE
    squared_diffs = (t_p - t_c) ** 2 #計算張量中每個元素的平方差值(**2 代表取平方)
    return squared_diffs.mean() #算出 squared_diffs 中所有元素之平均值


# In[4]:


w = torch.ones(()) #將 w 初始化為 1
b = torch.zeros(()) #將 b 初始化為 0
t_p = model(t_u, w, b)
t_p


# In[5]:


loss = loss_fn(t_p, t_c)
loss #輸出為均方損失，為一純量


# In[6]:


#broadcasting 範例
x = torch.ones(()) # x 是 0 軸張量
y = torch.ones(3, 1) # y 是 2 軸張量，shape 為 3 x 1
z = torch.ones(1, 3) # z 是 2 軸張量，shape 為 1 x 3
a = torch.ones(2, 1, 1) # a 是 3 軸張量，shape 為 2 x 1 x 1
print(f"shapes: x: {x.shape}, y: {y.shape}")
print(f"        z: {z.shape}, a: {a.shape}")
print("x * y:", (x * y).shape) #輸出 x * y 的 shape
print("y * z:", (y * z).shape) #輸出 y * z 的 shape
print("y * z * a:", (y * z * a).shape) #輸出 y * z * a 的 shape


# In[7]:


delta = 0.1

#損失對 w 的變化率 = ((w+delta)時的損失 - (w-delta)時的損失) / 2 x delta
loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)
loss_rate_of_change_w


# In[8]:


learning_rate = 1e-2 #設定學習率為 0.01

w = w - learning_rate * loss_rate_of_change_w # <- 剛剛算出來的損失變化率


# In[9]:


loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)

b = b - learning_rate * loss_rate_of_change_b


# In[10]:


#為了計算 dloss_fn / dw 使用連鎖率 拆分為 dloss_fn / dt_p x dt_p / dw
def dloss_fn(t_p, t_c): #計算 dloss_fn / dt_p
    dsq_diffs = 2 * (t_p - t_c) #依照公式計算導數
    return dsq_diffs / t_p.size(0) # <- t_p張量內的元素總數，由於要取平均，所以這裡需進行除法


# In[11]:


#定義 dt_p /dw
def dmodel_dw(t_u, w, b): #模型對 w 的導數
    return t_u

#定義 dt_p /db
def dmodel_db(t_u, w, b): #模型對 b 的導數
    return 1.0


# In[12]:


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()]) #將損失對 w 和 b 之導數堆疊在一起


# In[13]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_params = True):
    #n_epochs 為訓練次數，每跑完所有訓練資料一次為 1 個 epoch
    #params 包含參數 w 和 b 的 tuple
    
    for epoch in range(1, n_epochs + 1):
        w,b = params
        t_p = model(t_u, w, b) #運行模型
        loss = loss_fn(t_p, t_c) #計算損失
        grad = grad_fn(t_u, t_c, t_p, w, b) #計算梯度
        params = params - learning_rate * grad #更新參數
        print('Epoch %d: Loss %f' % (epoch, float(loss))) #顯示第 epoch 次訓練的損失
        if(print_params):
            print('\tParams: ', params) #印出參數值
            print('\tGrad: ', grad) #印出梯度
    return params


# In[14]:


training_loop(n_epochs = 100, #設定訓練次數為 100
             learning_rate = 1e-2, #設定學習率為 0.01
             params = torch.tensor([1.0, 0.0]), #設定 w 的初始值為 1，b 的初始值為 0
             t_u = t_u,
             t_c = t_c)

#參數 params 單次的更新幅度太大，使得最佳化的過程變得不穩定
#損失並沒有收斂到最小值，反而發散( diverge )了
#可藉由調整學習率來調整更新幅度，當訓練結果與期望不相符時，我們最常更改的參數之一便是學習率
#學習率的調整通常是以一次一個數量級(即每次乘以 10 或除以 10)


# In[15]:


training_loop(n_epochs = 100,
             learning_rate = 1e-4, #將學習率降低 100 倍
             params = torch.tensor([1.0, 0.0]),
             t_u = t_u,
             t_c = t_c)

#優化過程變的穩定了
#但是由於參數每次更新的幅度非常小，故損失下降的速度也非常緩慢，甚至最後停滯不前
#可透過 讓學習率隨著訓練次數改變 來解決
#根據訓練的進程，來相對應地調整學習率


# In[16]:


#在上個訓練的 epoch 1 中的 Grad，w 的梯度是 b 的 50倍左右
#這表示 w 和 b 是不同級數的
#在這種情況下，對某一參數來說大小適中的學習率，對另一個參數來說可能太大，進而導致無法對後者穩定的優化
#而對後者來說大小適中的學習率，則對前者過小，以至於沒辦法產生有效的參數更新

#透過 調整輸入資料 來降低梯度間的差異
#例如 我們可以讓輸入值的範圍大致落在 -1.0 ~ 1.0 之間 (該過程即為 正規化)

t_un = 0.1 * t_u #這樣做有類似正規化的效果

training_loop(n_epochs = 100,
             learning_rate = 1e-2, #學習率使用一開始造成梯度爆炸的數值
             params = torch.tensor([1.0, 0.0]),
             t_u = t_un, #原本的 t_u 替換成 t_un
             t_c = t_c)

#可以發現 w 和 b 的梯度量級差異變小，如此一來對兩者套運同一學習率也就沒問題


# In[17]:


params = training_loop(n_epochs = 5000,
             learning_rate = 1e-2,
             params = torch.tensor([1.0, 0.0]),
             t_u = t_un,
             t_c = t_c,
             print_params = False)

params


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

#*params 為 python 的引數解包(argument unpacking)技巧，因此 params 中的元素便會被當成單獨的引數傳入函式
# model(t_un, *params) => model(t_un, params[0], params[1])
#引數解包的應用對象通常是 list 或 tuple
t_p = model(t_un, *params) #<-為剛剛訓練過的參數。記住，訓練模型時使用的是經過正規化乘以 0.0 的輸入資料(t_un)

fig = plt.figure(dpi=100)
plt.xlabel("Temperature (Fahrenheit)") #設定 x 軸的名稱
plt.ylabel("Temperature (Celsius)") #設定 y 軸名稱
plt.plot(t_u.numpy(), t_p.detach().numpy()) #注意，我們畫的是未經正規化的資料

plt.plot(t_u.numpy(), t_c.numpy(), 'o')


# In[19]:


def model(t_u, w, b): #重新定義模型
    return w * t_u + b 

def loss_fn(t_p, t_c): #重新定義損失函數
    squared_diffs = (t_p - t_c) ** 2 
    return squared_diffs.mean() 


# In[30]:


#使用 requires_grad=True 後，PyTorch 會追蹤由 params 相關運算所產生的所有張量
#只要某張量的母張量中包 params，那麼從 params 到該張量之間所有的運算函數便會被記錄下來
#假設這些函數皆可微分，那其導數變會自動地存入 params 張量的 grad 屬性之中
params = torch.tensor([1.0, 0.0], requires_grad=True) #設定 w 的初始值為 1，b 的初始值為 0


# In[21]:


print(params.grad)


# In[31]:


loss = loss_fn(model(t_u, *params), t_c) #計算損失
loss.backward() #呼叫 loss 的 backward()，即開始反向傳播計算梯度
params.grad #透過 grad 屬性即可知道 params 中各參數的梯度 


# In[33]:


#呼叫 backward() 後，導數會被累加到葉節點的 grad 中
#假設我們先呼叫 backward() 一次、重新評估損失、然後再呼叫一次 backward()，則兩次呼叫 backward()的梯度會被相加，進而產生錯誤的結果
#為避免上述的問題，必須在每次訓練迴圈完成後，將梯度歸零
if params.grad is not None:
    params.grad.zero_()


# In[35]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_() #將梯度歸零
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward() #在呼叫 backward() 時，PyTorch 紀錄的前向運算圖會被清掉，只留下 params 葉節點(即 w 和 b節點)
        
        #PyTorch autograd 不會在 with torch.no_grad() 區塊中運作，因此會將 with 內的運算過程加到運算圖中
        #因此在下個回全建立新的前向運團前，不要更動到運算圖的內容，因此須將相關的城市放在 with torch.no_grad(): 中
        with torch.no_grad():
            params -= learning_rate * params.grad #利用梯度來更新參數
        if epoch % 500 == 0: #每隔 500 次訓練，就顯示一次損失值以追租訓練進度
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


# In[36]:


training_loop(n_epochs = 5000,
             learning_rate = 1e-2,
             params = torch.tensor([1.0, 0.0], requires_grad=True), #加入 requires_grad=True 是這裡的關鍵
             t_u = t_un, #和先前一樣使用正規化的 t_un 來取代 t_u
             t_c = t_c)


# In[37]:


import torch.optim as optim
dir(optim)


# In[40]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5

#每個優化器建構函式的第一個參數都是 內涵模型參數的張量，且該張量的 requires_grad 通常設為 True
#每個優化器有兩個 method: zero_grad() 和 step()，前者可以將所有參數張量的 grad 屬性歸零，後者則可以根據優化器自身的優化策略來更新數值
optimizer = optim.SGD([params], lr=learning_rate) #使用 SGD 優化器，並帶入(模型參數, 學習率)


# In[42]:


t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c) #計算損失
loss.backward() #反向傳播
optimizer.step() #更新參數
params


# In[44]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)
optimizer.zero_grad() #和之前一樣，此行程式的位置並非固定，只要在 loss.backward() 之前呼叫即可
loss.backward()
optimizer.step()
params


# In[46]:


def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0: #每 500 次訓練迴圈便印出一次損失
            print('Epoch %d, Loss %f' %(epoch, loss))
    return params


# In[47]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
training_loop(n_epochs = 5000,
             optimizer = optimizer,
             params = params,
             t_u = t_un, #此處使用正規化後的輸入
             t_c = t_c)


# In[51]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) #選擇 Adam 作為優化器
training_loop(n_epochs = 5000,
             optimizer = optimizer,
             params = params,
             t_u = t_u, #使用原始輸入 t_u (未經正規化)
             t_c = t_c)


# In[52]:


#拆分訓練集和驗證集
n_samples = t_u.shape[0] #取得輸入張量內的樣本總數
n_val = int(0.2 * n_samples) #設定驗證集內的樣本數為總樣本數的 20%
shuffled_indices = torch.randperm(n_samples) #產生隨機排列的索引值
train_indices = shuffled_indices[:-n_val] #取前 80% 作為訓練集的元素索引值
val_indices = shuffled_indices[-n_val:] #取後 20% 作為驗證集的元素索引值
train_indices, val_indices 


# In[53]:


train_t_u = t_u[train_indices] #建立訓練集
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices] #建立驗證集
val_t_c = t_c[val_indices] #建立驗證集

train_t_un = 0.1 * train_t_u #對輸入樣本進行正規化
val_t_un = 0.1 * val_t_u


# In[60]:


def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        with torch.no_grad(): #因為 這裡不會呼叫 v_loss.backward() 因此可以不用幫其計算運算圖，可用 torch.no_grad 來達成
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False #藉由檢查 val_loss 張量的 requires_grad 確定是否關閉 autograd
        
        optimizer.zero_grad()
        train_loss.backward() #注意，沒有 val_loss.backward()，因為我們不會用驗證資料來訓練模型
        optimizer.step()
        
        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epocch {epoch}, Training loss {train_loss.item():.4f},"
                 f"Validation loss {val_loss.item():.4f}")
    return params


# In[61]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(n_epochs = 3000,
              optimizer = optimizer,
              params = params,
              train_t_u = train_t_un, #因為此處用的優化器是 SGD，因此要用正規化的輸入資料
              val_t_u = val_t_un,
              train_t_c = train_t_c,
              val_t_c = val_t_c)


# In[62]:


def cal_forward(t_u, t_c, is_train):
    with torch.set_frad_enabled(is_train): #當 is_train = True 時，開啟 autograd
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss

