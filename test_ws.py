"""Test CoreML WS model with various sequence lengths."""

import coremltools as ct
import numpy as np
import torch
import json, os
from convert_ws import BertForTokenClassification, load_from_mlx_dir

print("Loading CoreML model...")
mlmodel = ct.models.MLModel("ckip_ws.mlpackage")

print("Loading PyTorch reference...")
pt_model, config = load_from_mlx_dir("../ckip_mlx/models/ws")
pt_model.eval()

# Load vocab
vocab = {}
with open("../ckip_mlx/models/ws/vocab.txt") as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i

def tokenize(text):
    ids = [vocab.get("[CLS]")] + [vocab.get(ch, vocab["[UNK]"]) for ch in text] + [vocab.get("[SEP]")]
    return ids

def decode_ws(preds, text):
    words, cur = [], ""
    for i, ch in enumerate(text):
        if preds[i+1] == 0 and cur:  # +1 for [CLS]
            words.append(cur); cur = ch
        else: cur += ch
    if cur: words.append(cur)
    return words

tests = [
    "台積電今天股價上漲",                    # 8 chars
    "中央研究院資訊科學研究所開發了自然語言處理工具。",  # 21 chars
    "台灣積體電路製造股份有限公司董事長劉德音今天在台北國際會議中心出席全球半導體論壇發表專題演講他表示台積電將在高雄楠梓科技園區投資超過一兆新台幣興建七奈米及二十八奈米晶圓廠",  # 80 chars
    "花蓮太魯閣國家公園的峽谷景觀聞名世界高雄港是台灣最大的國際商港每年處理數百萬個貨櫃故宮博物院收藏了超過六十萬件珍貴文物台灣半導體產業在全球供應鏈中扮演關鍵角色中華電信宣布五G網路覆蓋率已達百分之九十國立台灣大學醫學院附設醫院的急診室人滿為患他在台北市信義區的誠品書店買了三本小說總統府前舉行國慶大典數萬民眾到場觀禮" * 3,  # ~480 chars
]

print(f"\n{'len':>5} │ {'CoreML':>8} │ {'PyTorch':>8} │ Match │ WS Result")
print(f"{'─'*70}")

for text in tests:
    ids = tokenize(text)
    seq_len = len(ids)
    ids_np = np.array([ids], dtype=np.int32)
    mask_np = np.ones((1, seq_len), dtype=np.int32)

    # CoreML
    cm_out = mlmodel.predict({"input_ids": ids_np, "attention_mask": mask_np})
    cm_logits = list(cm_out.values())[0]
    cm_preds = np.argmax(cm_logits, axis=-1).flatten().tolist()

    # PyTorch
    with torch.no_grad():
        pt_out = pt_model(torch.tensor([ids]), torch.tensor([[1]*seq_len]))
    pt_preds = pt_out.argmax(-1).flatten().tolist()

    match = cm_preds == pt_preds
    ws = "｜".join(decode_ws(cm_preds, text))
    print(f"{len(text):>5} │ {seq_len:>8} │ {seq_len:>8} │ {'✓' if match else '✗':>5} │ {ws[:60]}")

print(f"\n動態長度測試完成！")
