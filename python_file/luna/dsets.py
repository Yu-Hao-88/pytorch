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
    'CandidateInfoTuple',  # 樣本 tuple 名字
    'isNodule_bool, diameter_mm, series_uid, center_xyz',  # 該 tuple 中包含的資訊
)


@functools.lru_cache(1)  # 標準函式庫中的記憶體內快取
def getCandidateInfoList(requireOnDisk_bool=True):
    # requireOnDisk_bool 預設為篩選掉資料子集中未就位的序列
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    # 由資料檔所在的路徑取得 mhd 檔案列表
    mhd_list = glob.glob('../../data/part2/luna_data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}  # 建立 結點字典 來儲存每個真結點的座標與直徑資訊
    with open('../../data/part2/luna/annotations.csv', "r") as f:  # 開啟 annotations.csv
        for row in list(csv.reader(f))[1:]:  # 跳過第一列
            series_uid = row[0]  # 取得第 0 行的資料 (代表 series UID)
            annotationCenter_xyz = tuple(
                [float(x) for x in row[1:4]])  # 將 xyz 座標整理成 tuple
            annotationDiameter_mm = float(row[4])  # 取得第 4 行的資料 (代表結點直徑大小)

            diameter_dict.setdefault(series_uid, []).append(
                # 儲存到結點字典中，key 為 series_uid，value 為以(xyz 座標，直徑大小)表示的 tuple
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []  # 創建一個串列來儲存候選節點的資訊
    with open('../../data/part2/luna/candidates.csv', "r") as f:  # 開啟 candidates.csv
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:  # 若找不到某 series_uid，那就代表其未被嚇在致硬碟中，應該跳過該筆資料
                continue

            isNodule_bool = bool(int(row[4]))  # 取得是否為結點的布林值
            candidateCenter_xyz = tuple([float(x)
                                        for x in row[1:4]])  # 取得 xyz 座標

            candidateDiameter_mm = 0.0
            # 處理 diameter_dict 中的樣本(真結點)
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):  # 依次走訪 x、y、z 座標
                    delta_mm = abs(
                        candidateCenter_xyz[i] - annotationCenter_xyz[i])  # 兩個 csv 中，擁有相同 series_uid 之樣本的中心點座標差距
                    # 檢查兩組中心點座標的差距是否大於節點直徑的四分之一(若是則是為不同節點)
                    if delta_mm > annotationDiameter_mm / 4:
                        break  # 若不符合條件，就跳過該筆資料
                else:
                    candidateDiameter_mm = annotationDiameter_mm  # 更新候選結點的大小
                    break

            candidateInfo_list.append(CandidateInfoTuple(  # 將資料存入 candidateInfo_list 串列中
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    # 在此串列中，真實的節點樣本會被排在前面(直徑由大到小)，非結點樣本(結點大小=0)則緊隨其後
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


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

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


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
            candidate_t = candidate_t.unsqueeze(0) # 在第 0 軸加入 通道 軸

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
