import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from util.logconf import logging
from model import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0  # 紀錄標籤的索引
METRICS_PRED_NDX = 1  # 紀錄預設結果的索引
METRICS_LOSS_NDX = 2  # 紀錄損失的索引
METRICS_SIZE = 3  # 表現指標的數量

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:  # 呼叫時沒有提供引數，則從命令列獲得引數
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )
        parser.add_argument('--balanced',  # 若將此參數加入命令列，代表要使用平衡機制
                            help="Balance the training data to half positive, half negative.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-flip',
                            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-offset',
                            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-scale',
                            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-noise',
                            help="Augment the training data by randomly adding noise to the data.",
                            action='store_true',
                            default=False,
                            )

        parser.add_argument('--tb-prefix',
                            default='log',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime(
            '%Y-%m-%d_%H.%M.%S')  # 這裡使用 timestamp 來顯示訓練相關資訊

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        # 根據經驗這些數值是有效的，但不排除有更好的選擇存在
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()  # 偵測電腦是否支援 CUDA
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.segmentation_model, self.augmentation_model = self.initModel()  # 初始化模型
        self.optimizer = self.initOptimizer()  # 初始化優化器

    def initModel(self):
        segmentation_model = UNetWrapper(
            # UNet 的輸入資料共有 7 個通道(in_channels=7)，分別代表 6 個提供脈絡訊息的切片以及 1 個實際被分割的目標切片
            in_channels=7,
            n_classes=1,  # 輸出的分類類別數量為1，用來說明某體素是否屬於結節的一部份
            depth=3,  # depth 用控制 U 型結構的深度:depth + 1 就會多加一層降採樣處理
            wf=4,  # wf=4 表示 model 第一層將有 2^wf==16 個過濾器(每經一次降採樣，此數量會翻一倍)
            padding=True,  # padding=True 表示要進行填補，如此一卷基層的輸出圖片大小才會與輸入相同
            batch_norm=True,  # batch_norm=True 表示在每個神經網路的激活函數後面，都要進行批次正規化
            up_mode='upconv',  # up_mode='upconv' 表示 model 的升採樣要使用升卷積模組
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

        # 在 13 章的時候沒使用
        # model = LunaModel()  # 創建一個 LunaModel 物件
        # if self.use_cuda:
        #     log.info("Using CUDA; {} devices.".format(
        #         torch.cuda.device_count()))
        #     if torch.cuda.device_count() > 1:  # 偵測是否具多個 GPU
        #         model = nn.DataParallel(model)  # 平行處理
        #     model = model.to(self.device)  # 將模型參數送往 GPU(或 CPU)
        # return model

    def initOptimizer(self):
        # 使用 SGD 優化器
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)
        # 將分割模型的參數傳進 Adam 優化器
        return Adam(self.segmentation_model.parameters())

    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    # def initTrainDl(self):
    #     train_ds = LunaDataset(  # 建立一個 LunaDataset 物件(train_ds)
    #         val_stride=10,
    #         isValSet_bool=False,  # 代表現在要建立的是訓練集
    #         # 若 balanced 為 True 則結果為1，因此陰陽樣本的比例為 1:1
    #         ratio_int=int(self.cli_args.balanced),
    #         augmentation_dict=self.augmentation_dict,  # 內存有各種資料擴增的策略
    #     )

    #     batch_size = self.cli_args.batch_size  # 在 precache.py 中的 Line 31 中已設定
    #     if self.use_cuda:
    #         batch_size *= torch.cuda.device_count()  # GPU 數量越大，batch_size 也越大

    #     train_dl = DataLoader(
    #         train_ds,
    #         batch_size=batch_size,  # 批次量
    #         num_workers=self.cli_args.num_workers,  # 運算單元的數量
    #         pin_memory=self.use_cuda,  # 鎖定記憶體 (pinned memory) 可以快速轉移至 GPU
    #     )

    #     return train_dl

    # def initValDl(self):
    #     val_ds = LunaDataset(  # 建立一個 LunaDataset 物件(train_ds)
    #         val_stride=10,
    #         isValSet_bool=True,  # 代表現在要建立的是驗證集
    #     )

    #     batch_size = self.cli_args.batch_size
    #     if self.use_cuda:
    #         batch_size *= torch.cuda.device_count()

    #     val_dl = DataLoader(
    #         val_ds,
    #         batch_size=batch_size,  # 批次量
    #         num_workers=self.cli_args.num_workers,  # 運算單元的數量
    #         pin_memory=self.use_cuda,  # 鎖定記憶體 (pinned memory) 可以快速轉移至 GPU
    #     )

    #     return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(
                'runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()  # 驗證用的資料載入器與訓練用的類似

        best_score = 0.0
        self.validation_cadence = 5  # 用來控制執行驗證的頻率

        for epoch_ndx in range(1, self.cli_args.epochs + 1):  # 代表當前的訓練週期

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)  # 執行一次訓練
            # 紀錄每個訓練週期後產生的表現評估指標
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            # 只在首輪以及往後的每 validation_cadence 輪訓練週期執行驗證
            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(
                    epoch_ndx, 'val', valMetrics_t)  # 計算模型分數(召回率)
                best_score = max(score, best_score)  # 若當前分數較高，則更新 best_score 值

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)  # 測試模型並記錄輸出圖片
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.segmentation_model.train()
        trnMetrics_g = torch.zeros(  # 初始化存放評估資訊的空陣列
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(  # 建立批次迴圈並計算時間
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()  # 將所有梯度歸零

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.trn_writer.close()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            # eval() 和 train() 的差異是他不會更新權重，而且還會關閉如 Dropout、batchnorm 等只有訓練時才需要的功能
            self.segmentation_model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(  # 計算驗證程序所需時間
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(  # 計算批次損失
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)  # 將張量移轉至 GPU
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:  # 只有在訓練時需要進行資料擴增，驗證時我們會跳過此步驟
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)  # 執行模型分割

        # 計算訓練樣本真正的 Dice loss
        diceLoss_g = self.diceLoss(prediction_g, label_g)
        # 只計算真陽性和偽陰性的 Dice Loss
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            # 將預測結果與特定閾值比較以取得硬性 Dice 係數(只由 0 或 1 組成)。
            # 為了之後能進行乘法，須將資料類型重新轉換為浮點數
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)

            # 計算真陽性、偽陽性、與偽陰性事件的數量(這裡和計算 Dice 損失類似)
            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])

            # 方便未來存取，將所有指標存於一大型張量中
            # 注意，這些結果是以批次為單位進行計算，而非取所有批次的平均值
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

         # *8 代表 找出所有陽性像素 比 找出所有陽性及陰性像素 重要 8 倍
        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    # 在 13 章中使用新的計算 loss 函式，故將舊的先註解
    # def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
    #     input_t, label_t, _series_list, _center_list = batch_tup  # 拆解批次 tuple

    #     input_g = input_t.to(self.device, non_blocking=True)  # 將張量移轉至 GPU
    #     label_g = label_t.to(self.device, non_blocking=True)

    #     logits_g, probability_g = self.segmentation_model(input_g)  # 運行模型

    #     # reduction='none' 讓我們可以計算 每個樣本 的損失
    #     loss_func = nn.CrossEntropyLoss(reduction='none')
    #     loss_g = loss_func(
    #         logits_g,
    #         label_g[:, 1],  # one-hot 編碼類別的所引
    #     )
    #     start_ndx = batch_ndx * batch_size
    #     end_ndx = start_ndx + label_t.size(0)

    #     metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
    #         label_g[:, 1].detach()  # 紀錄實際標籤
    #     metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
    #         probability_g[:, 1].detach()  # 紀錄預測結果
    #     metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
    #         loss_g.detach()  # 紀錄損失值

    #     return loss_g.mean()  # 將各樣本的損失結合成單一數值(計算平均)

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        # 對除了批次軸(第 0 軸)以外的所有維度取總和，以得到每批次中的陽性標註數、預測到的陽性數、以及真陽性數
        diceLabel_g = label_g.sum(dim=[1, 2, 3])
        dicePrediction_g = prediction_g.sum(dim=[1, 2, 3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)  # 計算 Dice 分數(為了避免意外未取得預測和標註的情況，分子和分母皆加上一個非常小的 epsilon 值)

        return 1 - diceRatio_g  # 為了符合損失的特性(越小越好)，此處傳回的結果為 1-Dice

    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()  # 將模型設成 eval 模式

        # 繞過資料匯入器，直接從資料集中選出 12 組 CT 資料
        # 由於儲存 series UID 的串列可能經過洗牌，故此處先進行排序
        images = sorted(dl.dataset.series_list)[:12]
        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid)

            for slice_ndx in range(6):
                # 選取該 CT 中 6張間距相等的切片
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                ct_t, label_t, series_uid, ct_ndx = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                # 將 CT 影像的強度指定給 RGB 三個通道，以產生灰階的底圖
                image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                # 偽陽性標記為紅色，然後貼於底圖上
                image_a[:, :, 0] += prediction_a & (1 - label_a)
                # 偽陰性標為橘色
                image_a[:, :, 0] += (1 - prediction_a) & label_a
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5
                # 真陽性則為綠色
                image_a[:, :, 1] += prediction_a & label_a
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                    # image_a[:,:,0] += (1 - label_a) & lung_a # Red
                    image_a[:, :, 1] += label_a  # Green
                    # image_a[:,:,2] += neg_a  # Blue

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC',  # 告訴 TensorBoard 輸出圖片的各軸中，RGB 通道位於最後一軸
                    )
                # This flush prevents TB from getting confused about which
                # data item belongs where.
                writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * \
            100  # 此處的值可能會超過 100%，因為我們是拿 偽陽性數 和 被標記成候選節點的像素總數(只占整章圖的一小部分)做比較

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value,
                              self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']  # 以召回率(recall)作為評分標準

        return score

    # 13章用了新的 logMetrics
    # def logMetrics(
    #         self,
    #         epoch_ndx,  # 本週期是第幾週期
    #         mode_str,  # 本週期是在訓練還是驗證
    #         metrics_t,  # 本週期的評估資訊
    #         classificationThreshold=0.5,  # 輸出機率大於此閾值時即視為陽性(是結點)
    # ):
    #     self.initTensorboardWriters()
    #     log.info("E{} {}".format(
    #         epoch_ndx,
    #         type(self).__name__,
    #     ))

    #     # 依照閾值製作陰性製作陰性標籤的遮罩，遮罩內為 True 的元素代表不是結點(陰性)，為 False 的則是結點(陽性)
    #     negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
    #     # 用同樣方法製作陰性預測值的遮罩
    #     negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

    #     posLabel_mask = ~negLabel_mask  # 製作陽性標籤的遮罩
    #     posPred_mask = ~negPred_mask  # 製作陽性預測值的遮罩

    #     neg_count = int(negLabel_mask.sum())  # 計算標籤中的陽性及陰性的數量
    #     pos_count = int(posLabel_mask.sum())

    #     trueNeg_count = neg_correct = int(
    #         (negLabel_mask & negPred_mask).sum())  # 計算真陰性(預測為陰性且正確)的數量
    #     truePos_count = pos_correct = int(
    #         (posLabel_mask & posPred_mask).sum())  # 計算真陽性(預測為陽性且正確)的數量

    #     falsePos_count = neg_count - neg_correct  # 偽陽性數
    #     falseNeg_count = pos_count - pos_correct  # 偽陰性數

    #     metrics_dict = {}
    #     metrics_dict['loss/all'] = \
    #         metrics_t[METRICS_LOSS_NDX].mean()  # 計算整體的平均損失
    #     metrics_dict['loss/neg'] = \
    #         metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()  # 計算陰性樣本的平均損失
    #     metrics_dict['loss/pos'] = \
    #         metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()  # 計算陽性樣本的平均損失

    #     metrics_dict['correct/all'] = (pos_correct + neg_correct) \
    #         / np.float32(metrics_t.shape[1]) * 100  # 計算整體準確率
    #     metrics_dict['correct/neg'] = neg_correct / \
    #         np.float32(neg_count) * 100  # 計算TN(真陰性率)
    #     metrics_dict['correct/pos'] = pos_correct / \
    #         np.float32(pos_count) * 100  # 計算TP(真陽性率)

    #     precision = metrics_dict['pr/precision'] = \
    #         truePos_count / \
    #         np.float32(truePos_count + falsePos_count)  # 精確率的公式
    #     recall = metrics_dict['pr/recall'] = \
    #         truePos_count / \
    #         np.float32(truePos_count + falseNeg_count)  # 召回率的公式

    #     metrics_dict['pr/f1_score'] = \
    #         2 * (precision * recall) / (precision +
    #                                     recall)  # 計算 F1 分數，並存入 metrics_dict

    #     log.info(
    #         ("E{} {:8} {loss/all:.4f} loss, "  # 整體損失
    #          + "{correct/all:-5.1f}% correct, "  # 整體準確率
    #          + "{pr/precision:.4f} precision, "  # 加入精準率
    #          + "{pr/recall:.4f} recall, "  # 加入召回率
    #          + "{pr/f1_score:.4f} f1 score"  # 加入 F1 分數
    #          ).format(
    #             epoch_ndx,
    #             mode_str,
    #             **metrics_dict,  # 從 metrics_dict 讀入各評估標準
    #         )
    #     )

    #     log.info(
    #         ("E{} {:8} {loss/neg:.4f} loss, "  # 陰性樣本的損失
    #          # 陰性樣本的分類準確率
    #          + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
    #          ).format(
    #             epoch_ndx,
    #             mode_str + '_neg',
    #             neg_correct=neg_correct,
    #             neg_count=neg_count,
    #             **metrics_dict,
    #         )
    #     )

    #     log.info(
    #         ("E{} {:8} {loss/pos:.4f} loss, "  # 陽性樣本的損失
    #          # 陽性樣本的分類準確率
    #          + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
    #          ).format(
    #             epoch_ndx,
    #             mode_str + '_pos',
    #             pos_correct=pos_correct,
    #             pos_count=pos_count,
    #             **metrics_dict,
    #         )
    #     )

    #     writer = getattr(self, mode_str + '_writer')

    #     for key, value in metrics_dict.items():
    #         writer.add_scalar(key, value, self.totalTrainingSamples_count)

    #     writer.add_pr_curve(
    #         'pr',
    #         metrics_t[METRICS_LABEL_NDX],
    #         metrics_t[METRICS_PRED_NDX],
    #         self.totalTrainingSamples_count,
    #     )

    #     bins = [x/50.0 for x in range(51)]

    #     negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
    #     posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

    #     if negHist_mask.any():
    #         writer.add_histogram(
    #             'is_neg',
    #             metrics_t[METRICS_PRED_NDX, negHist_mask],
    #             self.totalTrainingSamples_count,
    #             bins=bins,
    #         )
    #     if posHist_mask.any():
    #         writer.add_histogram(
    #             'is_pos',
    #             metrics_t[METRICS_PRED_NDX, posHist_mask],
    #             self.totalTrainingSamples_count,
    #             bins=bins,
    #         )

        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                     self.totalTrainingSamples_count,
    #                     # bins=bins,
    #                 )
    #             except Exception as e:
    #                 log.error([min_data, max_data])
    #                 raise

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module  # 去除 DataParallel 包裹器(如果存在的話)

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),  # 此行是關鍵
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),  # 保存諸如動量等超參數
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'data-unversioned', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    LunaTrainingApp().main()
