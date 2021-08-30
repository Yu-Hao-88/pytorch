import argparse
import datetime
import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dsets import LunaDataset
from util.logconf import logging
from model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0  # 紀錄標籤的索引
METRICS_PRED_NDX = 1  # 紀錄預設結果的索引
METRICS_LOSS_NDX = 2  # 紀錄損失的索引
METRICS_SIZE = 3  # 表現指標的數量


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

        self.model = self.initModel()  # 初始化模型
        self.optimizer = self.initOptimizer()  # 初始化優化器

    def initModel(self):
        model = LunaModel()  # 創建一個 LunaModel 物件
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:  # 偵測是否具多個 GPU
                model = nn.DataParallel(model)  # 平行處理
            model = model.to(self.device)  # 將模型參數送往 GPU(或 CPU)
        return model

    def initOptimizer(self):
        # 使用 SGD 優化器
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        train_ds = LunaDataset(  # 建立一個 LunaDataset 物件(train_ds)
            val_stride=10,
            isValSet_bool=False,  # 代表現在要建立的是訓練集
            # 若 balanced 為 True 則結果為1，因此陰陽樣本的比例為 1:1
            ratio_int=int(self.cli_args.balanced),
            augmentation_dict=self.augmentation_dict,  # 內存有各種資料擴增的策略
        )

        batch_size = self.cli_args.batch_size  # 在 precache.py 中的 Line 31 中已設定
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # GPU 數量越大，batch_size 也越大

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,  # 批次量
            num_workers=self.cli_args.num_workers,  # 運算單元的數量
            pin_memory=self.use_cuda,  # 鎖定記憶體 (pinned memory) 可以快速轉移至 GPU
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(  # 建立一個 LunaDataset 物件(train_ds)
            val_stride=10,
            isValSet_bool=True,  # 代表現在要建立的是驗證集
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,  # 批次量
            num_workers=self.cli_args.num_workers,  # 運算單元的數量
            pin_memory=self.use_cuda,  # 鎖定記憶體 (pinned memory) 可以快速轉移至 GPU
        )

        return val_dl

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

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
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
            self.model.eval()  # eval() 和 train() 的差異是他不會更新權重，而且還會關閉如 Dropout、batchnorm 等只有訓練時才需要的功能
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

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup  # 拆解批次 tuple

        input_g = input_t.to(self.device, non_blocking=True)  # 將張量移轉至 GPU
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)  # 運行模型

        # reduction='none' 讓我們可以計算 每個樣本 的損失
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1],  # one-hot 編碼類別的所引
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:, 1].detach()  # 紀錄實際標籤
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:, 1].detach()  # 紀錄預測結果
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()  # 紀錄損失值

        return loss_g.mean()  # 將各樣本的損失結合成單一數值(計算平均)

    def logMetrics(
            self,
            epoch_ndx,  # 本週期是第幾週期
            mode_str,  # 本週期是在訓練還是驗證
            metrics_t,  # 本週期的評估資訊
            classificationThreshold=0.5,  # 輸出機率大於此閾值時即視為陽性(是結點)
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        # 依照閾值製作陰性製作陰性標籤的遮罩，遮罩內為 True 的元素代表不是結點(陰性)，為 False 的則是結點(陽性)
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        # 用同樣方法製作陰性預測值的遮罩
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask  # 製作陽性標籤的遮罩
        posPred_mask = ~negPred_mask  # 製作陽性預測值的遮罩

        neg_count = int(negLabel_mask.sum())  # 計算標籤中的陽性及陰性的數量
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int(
            (negLabel_mask & negPred_mask).sum())  # 計算真陰性(預測為陰性且正確)的數量
        truePos_count = pos_correct = int(
            (posLabel_mask & posPred_mask).sum())  # 計算真陽性(預測為陽性且正確)的數量

        falsePos_count = neg_count - neg_correct  # 偽陽性數
        falseNeg_count = pos_count - pos_correct  # 偽陰性數

        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_t[METRICS_LOSS_NDX].mean()  # 計算整體的平均損失
        metrics_dict['loss/neg'] = \
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()  # 計算陰性樣本的平均損失
        metrics_dict['loss/pos'] = \
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()  # 計算陽性樣本的平均損失

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_t.shape[1]) * 100  # 計算整體準確率
        metrics_dict['correct/neg'] = neg_correct / \
            np.float32(neg_count) * 100  # 計算TN(真陰性率)
        metrics_dict['correct/pos'] = pos_correct / \
            np.float32(pos_count) * 100  # 計算TP(真陽性率)

        precision = metrics_dict['pr/precision'] = \
            truePos_count / \
            np.float32(truePos_count + falsePos_count)  # 精確率的公式
        recall = metrics_dict['pr/recall'] = \
            truePos_count / \
            np.float32(truePos_count + falseNeg_count)  # 召回率的公式

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision +
                                        recall)  # 計算 F1 分數，並存入 metrics_dict

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "  # 整體損失
             + "{correct/all:-5.1f}% correct, "  # 整體準確率
             + "{pr/precision:.4f} precision, "  # 加入精準率
             + "{pr/recall:.4f} recall, "  # 加入召回率
             + "{pr/f1_score:.4f} f1 score"  # 加入 F1 分數
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,  # 從 metrics_dict 讀入各評估標準
            )
        )

        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "  # 陰性樣本的損失
             # 陰性樣本的分類準確率
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )

        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "  # 陽性樣本的損失
             # 陽性樣本的分類準確率
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

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


if __name__ == '__main__':
    LunaTrainingApp().main()
