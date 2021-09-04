import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('luna_data_raw')

# 此 tuple 用來儲存 經過整理的人為標註資料
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

# CandidateInfoTuple = namedtuple(
#     'CandidateInfoTuple',  # 樣本 tuple 名字
#     'isNodule_bool, diameter_mm, series_uid, center_xyz',  # 該 tuple 中包含的資訊
# )


@functools.lru_cache(1)  # 標準函式庫中的記憶體內快取
def getCandidateInfoList(requireOnDisk_bool=True):
    # requireOnDisk_bool 預設為篩選掉資料子集中未就位的序列
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    # 由資料檔所在的路徑取得 mhd 檔案列表
    mhd_list = glob.glob('../../data/part2/luna_data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # 在第13章不被使用
    # diameter_dict = {}  # 建立 結點字典 來儲存每個真結點的座標與直徑資訊
    # with open('../../data/part2/luna/annotations.csv', "r") as f:  # 開啟 annotations.csv
    #     for row in list(csv.reader(f))[1:]:  # 跳過第一列
    #         series_uid = row[0]  # 取得第 0 行的資料 (代表 series UID)
    #         annotationCenter_xyz = tuple(
    #             [float(x) for x in row[1:4]])  # 將 xyz 座標整理成 tuple
    #         annotationDiameter_mm = float(row[4])  # 取得第 4 行的資料 (代表結點直徑大小)

    #         diameter_dict.setdefault(series_uid, []).append(
    #             # 儲存到結點字典中，key 為 series_uid，value 為以(xyz 座標，直徑大小)表示的 tuple
    #             (annotationCenter_xyz, annotationDiameter_mm)
    #         )

    # candidateInfo_list = []  # 創建一個串列來儲存候選節點的資訊
    # with open('../../data/part2/luna/candidates.csv', "r") as f:  # 開啟 candidates.csv
    #     for row in list(csv.reader(f))[1:]:
    #         series_uid = row[0]

    #         if series_uid not in presentOnDisk_set and requireOnDisk_bool:  # 若找不到某 series_uid，那就代表其未被嚇在致硬碟中，應該跳過該筆資料
    #             continue

    #         isNodule_bool = bool(int(row[4]))  # 取得是否為結點的布林值
    #         candidateCenter_xyz = tuple([float(x)
    #                                     for x in row[1:4]])  # 取得 xyz 座標

    #         candidateDiameter_mm = 0.0
    #         # 處理 diameter_dict 中的樣本(真結點)
    #         for annotation_tup in diameter_dict.get(series_uid, []):
    #             annotationCenter_xyz, annotationDiameter_mm = annotation_tup
    #             for i in range(3):  # 依次走訪 x、y、z 座標
    #                 delta_mm = abs(
    #                     candidateCenter_xyz[i] - annotationCenter_xyz[i])  # 兩個 csv 中，擁有相同 series_uid 之樣本的中心點座標差距
    #                 # 檢查兩組中心點座標的差距是否大於節點直徑的四分之一(若是則是為不同節點)
    #                 if delta_mm > annotationDiameter_mm / 4:
    #                     break  # 若不符合條件，就跳過該筆資料
    #             else:
    #                 candidateDiameter_mm = annotationDiameter_mm  # 更新候選結點的大小
    #                 break

    #         candidateInfo_list.append(CandidateInfoTuple(  # 將資料存入 candidateInfo_list 串列中
    #             isNodule_bool,
    #             candidateDiameter_mm,
    #             series_uid,
    #             candidateCenter_xyz,
    #         ))

    candidateInfo_list = []  # 新的節點資料串列
    with open('../../data/part2/luna/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:  # 將檔案中的每一行的標註資料取出來
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            # 作者已在此 csv 檔加入某結節是否為惡性的欄位，在此也一併儲存以供下一章使用
            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidateInfo_list.append(  # 將各行的標注資料加入串列中
                CandidateInfoTuple(
                    True,  # 將 isNodule_bool(是否為結節) 設為 True
                    True,  # 將 hasAnnotation_bool(是否有標註) 設為 True
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

    with open('../../data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:  # 將檔案中的每一行的標註資料取出來
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:  # 只存去其中和結節無關的項目(結節的項目以在上方取得)
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,  # isNodule_bool 設為 False(顯示是否為結節的布林值)
                        False,  # hasAnnotation_bool 射為 False(顯示是否有標註的布林值)
                        False,  # isMal_bool 設為 False(顯示是否為惡性的布林值)
                        0.0,  # 因為不是結節，因此直徑大小設為 0
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    # 在此串列中，真實的節點樣本會被排在前面(直徑由大到小)，非結點樣本(結點大小=0)則緊隨其後
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,  # (1)先取出字典中 key 為 series UID 的傳列，若 key 不存在則新增一個值為空白串列
                                      []).append(candidateInfo_tup)  # (2)再將候選結節加入此串列中

    return candidateInfo_dict


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            '../../data/part2/luna_data/subset*/{}.mhd'.format(series_uid)
        )[0]  # 在所有以 subset 開頭的子資料夾中找出名為指定 series_uid 的檔案

        ct_mhd = sitk.ReadImage(mhd_path)  # 使用 SimpleITK 來匯入資料
        # 因為要將數值轉換為 np.float32 格式，故需重新建立一個 np.array
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        # 清理資料，將 ct_a 的值限制在 -1000HU 到 1000HU 之間，範圍以外的離群值與最終目標無關，卻會增加完成任務的難易度
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())  # 取得原點的偏移量
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())  # 取得體素的大小資訊
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(
            3, 3)  # 將方向轉換成陣列，並將內含 9 個元素的陣列重塑成 3x3 矩陣

        candidateInfo_list = getCandidateInfoDict()[
            self.series_uid]  # 取得候選結節的串列

        self.positiveInfo_list = [  # 創建包含真實結節的串列
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool  # 代表是真正的結節
        ]
        self.positive_mask = self.buildAnnotationMask(
            self.positiveInfo_list)  # 建立遮罩
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2))  # 傳回一個 1D 向量，其中紀錄在此切片中，有多少遮罩體素被標記為結節
                                 .nonzero()[0].tolist())  # 將標記體素數不為零的遮罩切片索引值存於串列中

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):  # 密度閾值
        # 建立與 CT 資料一樣大的布林張量(用來存放遮罩)，其中所有值初始化為 False
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidateInfo_tup in positiveInfo_list:  # 走訪 positiveInfo_list 中每個結節的位置
            center_irc = xyz2irc(  # 先將 XYZ 座標轉換成 IRC 座標
                candidateInfo_tup.center_xyz,  # 這裡的 candidateInfo_tup 由 getCandidateInfoList 回傳
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)  # 取得搜索起點(即邊界框中心)的索引位置
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:  # 搜索邊界框
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:  # thershold 為我們設定的閾值
                    index_radius += 1
            except IndexError:  # 為避免張量索引超過張量大小所做的預防措施
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[  # 依結節的中心及3個軸的半徑，將半徑區域內的位置都標記為 True，以作為遮罩
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)  # 只保留遮罩中密度高於閾值的體素

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):  # 傳入 中心座標、以體素為單位的寬度
        center_irc = xyz2irc(  # 先將中心座標改成以 IRC 座標系統來表示
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []  # 存放進行切塊的位置
        for axis, center_val in enumerate(center_irc):  # 依序走訪 I、R、C 3個軸
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))  # 取得 CT 切塊

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz,
                                                         width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)  # 將 CT 體素值限縮再 -1000~1000 之間
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    # 此 CT 掃描中的切片數量, 陽性切片的索引列表
    return int(ct.hu_a.shape[0]), ct.positive_indexes


def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
    if use_cache:
        ct_chunk, center_irc = \
            getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)  # 創建一個 4x4 的張量，其中只對對角線的元素值為 1，其餘皆為 0
    # ... <1>

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:  # 若隨機產生的數字>0.5，就對轉換矩陣中的特定元素取負號
                # 要將資料做鏡像翻轉，只需將轉換矩陣中相關的元素加上一個負號即可
                transform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2  # 隨機產生一個旋轉角度
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(  # 用 PyTorch 做仿射變換
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False,
    )

    augmented_chunk = F.grid_sample(  # 用 PyTorch 做再取樣
        ct_t,
        affine_t,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=10,  # 設定存入驗證集的頻率
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,  # ratio_int 的預設值為 0，代表不進行平衡
                 augmentation_dict=None,
                 candidateInfo_list=None,
                 ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        # 將傳回值複製一份，這樣即使改變 candidateInfo_list 也不會影響原來的資料
        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:  # 當 isValSet_bool 為 True 時，建立驗證集
            assert val_stride > 0, val_stride
            # 每隔 10 筆資料，便從 candidateInfo_list 中取出一筆資料放入驗證集
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(
                key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [  # 創建陰性樣本串列
            nt for nt in self.candidateInfo_list if not nt.isNodule_bool  # 若 True 則代表是結點
        ]
        self.pos_list = [  # 創建陽性樣本串列
            nt for nt in self.candidateInfo_list if nt.isNodule_bool
        ]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):  # 在每個週期開始時都會呼叫此方法來打亂樣本串列順序
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.ratio_int:
            return 200000
        else:
            return len(self.candidateInfo_list)  # 傳回候選串列中的樣本數

    def __getitem__(self, ndx):
        if self.ratio_int:  # 若 ratio_int > 0，就會進行資料平衡
            pos_ndx = ndx // (self.ratio_int + 1)  # 取得陽性樣本串列中的索引

            if ndx % (self.ratio_int + 1):  # 若餘數不為零，代表此處應從陰性樣本串列中取出樣本，並放進新的資料集中
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)  # 若索引值超過串列長度，則取餘數
                candidateInfo_tup = self.negative_list[neg_ndx]  # 從陰性樣本串列中取樣本
            else:
                pos_ndx %= len(self.pos_list)  # 若索引值超過串列長度，取餘數
                candidateInfo_tup = self.pos_list[pos_ndx]  # 從陽性樣本中取得樣本
        else:  # 若 ratio_int 等於零，則代表不進行資料平衡
            # 若不進行平衡處理，則直接取原始資料集中的第 ndx 個樣本
            candidateInfo_tup = self.candidateInfo_list[ndx]

        width_irc = (32, 48, 48)  # 設定要取的資料尺寸

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)  # 在第 0 軸加入 通道 軸

        pos_t = torch.tensor([
            # 若是結點 pos_t 為[0, 1]，否則為[1, 0]
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,  # 這就是訓練用樣本
            torch.tensor(center_irc),
        )


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
                 ):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:  # 驗證階段
            assert val_stride > 0, val_stride
            # 由 series UID 串列的索引 0 開始，每隔 val_stride 便取 1 個元素用於驗證
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:  # 訓練階段
            # 每個 val_stride 便將 1 個元素刪除，剩下的便是訓練資料
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]  # 將所有切片的索引都加入 sample_list 中
            else:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]  # 只將陽性切片的索引加入 sample_list 中

        self.candidateInfo_list = getCandidateInfoList()  # 從快取中取得資料

        series_set = set(self.series_list)  # 建立集合以提升搜尋速度
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]  # 將不在上述集合中的候選樣本過濾掉

        self.pos_list = [nt for nt in self.candidateInfo_list
                         if nt.isNodule_bool]  # 在進行資料平衡以前，先取得由真實節點樣本組成的串列

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation',
                False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(
            self.sample_list)]  # 取餘數以確保索引在範圍內
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        # contextSlices_count 計算傳回的資料中共有多少切片
        ct_t = torch.zeros((self.contextSlices_count * 2 +
                           1, 512, 512))  # 配置可儲存多張切片的空間

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)  # 當超過 ct_a 的邊界時，複製第一或最後一個切片
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        # clamp() 和 clip() 的功能類似，可將所有體素值限縮在 -1000~1000之間
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(
            ct.positive_mask[slice_ndx]).unsqueeze(0)  # 取得該切片的陽性遮罩

        return ct_t, pos_t, ct.series_uid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)  # 進行裁切以升成訓練資料

    def getitem_trainingCrop(self, candidateInfo_tup):
        # 利用 candidateInfo_tup 中的資訊取得候選樣本所在的 7 張 96x96 的切片(ct_a) 與遮罩(pos_a)，以及結節的中位置(center_irc)
        ct_a, pos_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96),
        )
        pos_a = pos_a[3:4]  # 取出中間那張遮罩切片，以做為訓練的標籤

        # 在 0 到 31 之間隨機挑兩個數，作為 64x64 裁切框的左上角位置
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64,  # 對 CT 資料進行裁切
                                     col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64,  # 對遮罩資料進行裁切
                                       col_offset:col_offset+64]).to(torch.long)

        slice_ndx = center_irc.index  # 取得中心切片在 CT 資料中的索引位置

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx


class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()
        self.pos_list = [
            nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtRawCandidate(candidateInfo_tup.series_uid,
                          candidateInfo_tup.center_xyz, (7, 96, 96))

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1  # candidate_t, pos_t, series_uid, center_t
