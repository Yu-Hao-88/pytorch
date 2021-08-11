# pytorch

## 初始化
建立 conda env 和 activate:
```bash=
conda create --name pytorch python=3.6.8
conda avtivate pytorch
#移除環境
conda env remove --name pytorch
```
安裝套件:
```bash=
pip install -r requirements.txt

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

conda install h5py
```

開啟 Jupyter Notebook:
```bash=
Jupyter Notebook
```

## pytorch 常用函式庫
- 建構神經網路的核心模組位於 torch.nn 中，常見的有:
    - 完全連接層( fully connected layers )
    - 卷積層( convolutional layers )
    - 激活函數( activation functions )
    - 損失函數( loss functions )
- 載入和處理資料時，torch.util.data 是個好用的工具，主要有兩個類別(class)分別為:
    - Dataset，是連接使用者資料(任何格式的資料)和標準 PyTorch 張量的橋樑
    - DataLoader，他可以在後台生成子程序( child process )，並從資料集中在入資料
- 使用多個GPU或者讓多台機器同時訓練模型，可用 torch.nn.parallel.DistributedDataParallel 和 torch.distributed 來利用額外的硬體資源
- torch.optim 提供了優化模型的標準方法
- dtype 總覽:
    - torch.float32 or torch.float:32位元的單精度浮點數 (預設
    - torch.float64 or torch.double:64位元的雙精度浮點數
    - torch.float16 or torch.half:16位元的半精度浮點數
    - torch.int8:8位元的整數
    - torch.uint8:8位元的正整數
    - torch.int16 or torch.short:16位元的整數
    - torch.int32 or torch.int:32位元的整數
    - torch.int64 or torch.long:64位元的整數
    - torch.bool:布林值
- 常見 tensor 的操作:
    - Creation:用來創建張量的函式，ex: zeros(), ones()
    - Indexing、Slicing:用來改變張量的 shape 或內容的函式，ex: transpose()
    - Math:可以對張量數值進行運算的函式
        - Pointwise:對每一個元素進行轉換，並得到一個新張量，ex: abs()、cos()
        - Reduction:以迭代的方式對多個元素進行運算，ex: mean()、std()、norm()
        - Comparision:對張量內的元素值進行比較，ex: equal()、max()
        - Spectral:用來在頻域(frequency domain)和時域(time domain)中進行轉換和操作的函式，ex: stft()、hamming_window()
        - BLAS、LAPACK:BLAS代表基礎線性代數程式集(Basic Linear Algebra Subprograms)，LAPACK則代表線性代數套件(Linear Algebra PACKage)，它們專門用來處理純量、向量及陣列間的操作
        - 其他操作:特殊用途的函式，ex: 針對向量的cross()、針對矩陣的trace()
    - Random sampling:透過不同的機率分布進行隨機取樣，可用來生成亂數，ex: randn()、normal()
    - Serialization:用來讀取與儲存張量的函式，ex: load()、save()
    - Parallelism:在CPU的平行處理中，用來控制執行敘數量的函式，ex: set_num_threads()
    - 函式若結尾有 _ ，表示該方法為 in-place 操作，會直接修改記憶體內的值，而不是複製一份原始資料，處理後再傳回新張量
## 筆記
- 利用深度學習來完成任務需要的條件:
    1. 找到能處理輸入資料的方法
    2. 定義深度學習機器(或稱為模型)
    3. 找到能進行訓練以及萃取特徵，並讓行輸出正確答案的自動化方法
- 為了訓練模型需要準備的東西:
    1. 一組訓練資料集
    2. 一個可根據【訓練資料集】去調整【模型參數】的優化器
    3. 一個將模型和資料與硬體整合，並利用硬體來進行訓練的方法
- 張量:
    - 張量中的數值被置放於連續的記憶體區塊中，以一個 storage 為一個幾本儲存單位
    - 一個 storage 儲存了包含數值資料的 1軸連續陣列，可為 float32、int64 ......
    - 張量是用來呈現 storage 的視圖(view)，它讓我們能依照偏移量(offset)和步長(stride)來索引 storage 的內容
    - 張量能用多種方式對同一個 storage 進行索引，進而產生不同的張量，ex: 對同為1維陣列的 storage，分別以 (3, 2) (2, 3) 等不同 shape 來取用它，則會產生不同的 shape 的 tensoe 
- 要對 storage 進行索引，張量必須一類一些和 storage 相關的資訊:
    - 大小(size)(在 NumPy 中稱為 shape):
        - 是一個 tuple，表示張量在軸上有多少元素
        - ex: (列, 行) => (2, 3)，表示 有2列，每一列3個元素
    - 偏移(offset)
        - 是張量中首個元素在 storage 中的索引值，預設是 0
        - ex: offset=1，則張量的首個元素在 storage 中索引為 1
    - 步長(stride)
        - 是在提取各軸的下一個元素時，需要跳過的元素數量
        - ex: stride = (3, 1)，表示 +3 => next row，+1 => next col
    - 要知道索引(i, j)的元素在 storage 中的索引值，可透過以下取得:
    ```python
    storage_offset + stride[0] * i + stride[1] * j
    ```
    - 張量和 storage 的間接轉換讓一些操作(ex: 轉置張量或提取子張量)變得方便，因為記憶體無需重新分配，透過修改大小、偏移、步長產生一個新的張量
    - 轉置的概念就是 步長 x,y => y,x 交換
- PyTorch 與 NumPy 的互通性:
    - 使用 numpy() 轉換後會回傳一個與 points 相同 shape 和內容的 Numpy 多維陣列
    - 這個 NumPy 陣列會跟張量共用同一個底層記憶體
    - 若修改了 NumPy 陣列中的內容，原本的張量內容也會一起修改
    - 若張量是存放在 GPU 中，PyTorch 會把內容複製一分到 CPU，並把資料型別轉換為 NumPy  陣列
    - PyTorch 張量預設型別為 32位元浮點數，而 NumPy 則是 64位元浮點數，引此轉換後要留意 dtype
- PyTorch 要求圖片資料張量的軸順序為 **N(圖片張數 batch) X C(色彩通道) X H(高度) X W(寬度)**