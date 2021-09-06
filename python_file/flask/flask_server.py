import numpy as np
import sys
import os
import torch
from flask import Flask, request, jsonify
import json

from luna.model import LunaModel

app = Flask(__name__)

model = LunaModel()  # 建立模型
model.load_state_dict(torch.load(sys.argv[1],
                                 map_location='cpu')['model_state'])  # 匯入權重
model.eval()  # 開啟eval 模式（評估模式）


def run_inference(in_tensor):
    with torch.no_grad():  # 這裡不需要計算梯度
        # LunaModel takes a batch and outputs a tuple (scores, probs)
        out_tensor = model(in_tensor.unsqueeze(0))[1].squeeze(0)  # 運行模型
    probs = out_tensor.tolist()
    out = {'prob_malignant': probs[1]}
    return out


# 我們預期在『/predict』端點收到經由HTTP POST 傳來的表單資料
@app.route("/predict", methods=["POST"])
def predict():
    meta = json.load(request.files['meta'])
    blob = request.files['blob'].read()
    in_tensor = torch.from_numpy(np.frombuffer(
        blob, dtype=np.float32))  # 將二進位資料轉換為torch 張量
    in_tensor = in_tensor.view(*meta['shape'])
    out = run_inference(in_tensor)
    return jsonify(out)  # 傳回JSON 格式的預測結果


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    print(sys.argv[1])
