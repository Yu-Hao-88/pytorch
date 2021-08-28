import copy
import csv
import functools
import glob
import os

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('luna_data_raw')

#此 tuple 用來儲存 經過整理的人為標註資料
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', #樣本 tuple 名字
    'isNodule_bool, diameter_mm, series_uid, center_xyz', #該 tuple 中包含的資訊
)


@functools.lru_cache(1) #標準函式庫中的記憶體內快取
def getCandidateInfoList(requireOnDisk_bool=True): 
    #requireOnDisk_bool 預設為篩選掉資料子集中未就位的序列
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('../../data/part2/luna_data/subset*/*.mhd') #由資料檔所在的路徑取得 mhd 檔案列表
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {} #建立 結點字典 來儲存每個真結點的座標與直徑資訊
    with open('../../data/part2/luna/annotations.csv', "r") as f: #開啟 annotations.csv
        for row in list(csv.reader(f))[1:]: #跳過第一列
            series_uid = row[0] #取得第 0 行的資料 (代表 series UID)
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]]) #將 xyz 座標整理成 tuple
            annotationDiameter_mm = float(row[4]) #取得第 4 行的資料 (代表結點直徑大小)

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm) #儲存到結點字典中，key 為 series_uid，value 為以(xyz 座標，直徑大小)表示的 tuple
            )

    candidateInfo_list = [] #創建一個串列來儲存候選節點的資訊
    with open('../../data/part2/luna/candidates.csv', "r") as f: #開啟 candidates.csv
        for row in list(csv.reader(f))[1:]: 
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool: #若找不到某 series_uid，那就代表其未被嚇在致硬碟中，應該跳過該筆資料
                continue

            isNodule_bool = bool(int(row[4])) #取得是否為結點的布林值
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]]) #取得 xyz 座標

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []): #處理 diameter_dict 中的樣本(真結點)
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3): #依次走訪 x、y、z 座標
                    delta_mm = abs(
                        candidateCenter_xyz[i] - annotationCenter_xyz[i]) #兩個 csv 中，擁有相同 series_uid 之樣本的中心點座標差距
                    if delta_mm > annotationDiameter_mm / 4: #檢查兩組中心點座標的差距是否大於節點直徑的四分之一(若是則是為不同節點)
                        break #若不符合條件，就跳過該筆資料
                else:
                    candidateDiameter_mm = annotationDiameter_mm #更新候選結點的大小
                    break

            candidateInfo_list.append(CandidateInfoTuple( #將資料存入 candidateInfo_list 串列中
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True) #在此串列中，真實的節點樣本會被排在前面(直徑由大到小)，非結點樣本(結點大小=0)則緊隨其後
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            '../../data/part2/luna_data/subset*/{}.mhd'.format(series_uid)
        )[0] #在所有以 subset 開頭的子資料夾中找出名為指定 series_uid 的檔案

        ct_mhd = sitk.ReadImage(mhd_path) #使用 SimpleITK 來匯入資料
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32) #因為要將數值轉換為 np.float32 格式，故需重新建立一個 np.array

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a) #清理資料，將 ct_a 的值限制在 -1000HU 到 1000HU 之間，範圍以外的離群值與最終目標無關，卻會增加完成任務的難易度

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin()) #取得原點的偏移量
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing()) #取得體素的大小資訊
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3) #將方向轉換成陣列，並將內含 9 個元素的陣列重塑成 3x3 矩陣

    def getRawCandidate(self, center_xyz, width_irc): #傳入 中心座標、以體素為單位的寬度
        center_irc = xyz2irc( #先將中心座標改成以 IRC 座標系統來表示
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = [] #存放進行切塊的位置
        for axis, center_val in enumerate(center_irc): #依序走訪 I、R、C 3個軸
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

            slice_list.append(slice(start_ndx, end_ndx)) #取得 CT 切塊

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


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=10, #設定存入驗證集的頻率
                 isValSet_bool=None,
                 series_uid=None,
                 ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList()) #將傳回值複製一份，這樣即使改變 candidateInfo_list 也不會影響原來的資料

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool: #當 isValSet_bool 為 True 時，建立驗證集
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride] #每隔 10 筆資料，便從 candidateInfo_list 中取出一筆資料放入驗證集
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list) #傳回候選串列中的樣本數

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48) #設定要取的資料尺寸

        candidate_a, center_irc = getCtRawCandidate( #其傳回值 candidate_a 的 shape 為 (32, 48, 48)，分別為所引數(即切分片數量)、列數及行數
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0) #在第 0 軸加入 通道 軸

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool, #若是結點 pos_t 為[0, 1]，否則為[1, 0]
            candidateInfo_tup.isNodule_bool
        ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid, #這就是訓練用樣本
            torch.tensor(center_irc),
        )
