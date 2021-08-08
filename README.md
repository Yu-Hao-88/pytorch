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

## 筆記
- 利用深度學習來完成任務需要的條件:
    1. 找到能處理輸入資料的方法
    2. 定義深度學習機器(或稱為模型)
    3. 找到能進行訓練以及萃取特徵，並讓行輸出正確答案的自動化方法
- 為了訓練模型需要準備的東西:
    1. 一組訓練資料集
    2. 一個可根據【訓練資料集】去調整【模型參數】的優化器
    3. 一個將模型和資料與硬體整合，並利用硬體來進行訓練的方法

