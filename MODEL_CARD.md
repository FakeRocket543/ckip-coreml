---
language:
  - zh
license: gpl-3.0
library_name: coremltools
tags:
  - coreml
  - bert
  - token-classification
  - word-segmentation
  - pos-tagging
  - named-entity-recognition
  - traditional-chinese
  - ckip
  - apple-neural-engine
  - ios
datasets:
  - ckiplab/ckip-transformers
base_model:
  - ckiplab/bert-base-chinese-ws
  - ckiplab/bert-base-chinese-pos
  - ckiplab/bert-base-chinese-ner
---

# CKIP BERT-base CoreML — 繁體中文 WS/POS/NER for iOS/macOS

Apple CoreML 版本的 CKIP BERT-base 繁體中文 NLP 模型，可在 iOS/macOS 上透過 Apple Neural Engine (ANE) 執行。

- 📦 原始碼：[GitHub — FakeRocket543/ckip-coreml](https://github.com/FakeRocket543/ckip-coreml)
- 🤗 模型權重：[HuggingFace — FakeRockert543/ckip-coreml](https://huggingface.co/FakeRockert543/ckip-coreml)（本頁）
- 🧪 MLX 版本（桌面推薦）：[HuggingFace — FakeRockert543/ckip-mlx](https://huggingface.co/FakeRockert543/ckip-mlx)

從 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 轉換而來。CoreML fp16 是所有框架中最快的，比 CKIP 官方快 **6.3 倍**。

## 模型說明

| 任務 | 說明 | 標籤數 | 原始模型 |
|------|------|------:|---------|
| WS | 中文斷詞 | 2 (B/I) | ckiplab/bert-base-chinese-ws |
| POS | 詞性標注 | 60 | ckiplab/bert-base-chinese-pos |
| NER | 命名實體辨識 | 73 (BIOES) | ckiplab/bert-base-chinese-ner |

所有模型支援動態序列長度 1–512。

## 可用版本

| 版本 | 單模型大小 | 精度 (vs fp32) | 建議用途 |
|------|--------:|--------------|---------|
| fp32 | 388 MB | baseline (與 MLX fp32 100% 一致) | 追求完全精度 |
| fp16 | 194 MB | WS 100% / POS 99.97% / NER 99.99% | **推薦預設** ⚡ |
| q8 | 98 MB | WS 99.96% / POS 98.83% / NER 99.76% | 低記憶體 iPhone |

## 從零開始使用（完整步驟）

### 1. 環境準備

需要 macOS + Apple Silicon。CoreML Tools 目前需要 Python 3.13（不支援 3.14）。

```bash
# 取得原始碼
git clone https://github.com/FakeRocket543/ckip-coreml.git
cd ckip-coreml

# 建立虛擬環境（需要 Python 3.13）
python3.13 -m venv .venv && source .venv/bin/activate

# 安裝依賴
pip install coremltools numpy huggingface_hub
```

### 2. 下載模型

```bash
# 從 HuggingFace 下載全部 .mlpackage
huggingface-cli download FakeRockert543/ckip-coreml --local-dir .
```

下載後目錄結構：

```
ckip-coreml/
├── ckip_ws_fp16.mlpackage      # 斷詞 fp16（推薦）
├── ckip_ws_fp32.mlpackage
├── ckip_ws_q8.mlpackage
├── ckip_pos_fp16.mlpackage     # 詞性 fp16
├── ckip_pos_fp32.mlpackage
├── ckip_pos_q8.mlpackage
├── ckip_ner_fp16.mlpackage     # 實體 fp16
├── ckip_ner_fp32.mlpackage
├── ckip_ner_q8.mlpackage
├── ckip_ws.mlpackage           # 原始版本 (=fp32)
├── ckip_pos.mlpackage
└── ckip_ner.mlpackage
```

### 3. 準備詞表

CoreML 模型不包含詞表，需要從 MLX 版本取得，或自行下載 BERT 中文詞表：

```bash
# 方法一：從 MLX repo 下載 vocab.txt
huggingface-cli download FakeRockert543/ckip-mlx models/vocab.txt --local-dir .
mv models/vocab.txt vocab.txt && rm -rf models

# 方法二：從原始 BERT 下載
# wget https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt
```

### 4. 執行斷詞（Python）

```python
import coremltools as ct
import numpy as np

# 載入詞表
vocab = {}
with open("vocab.txt") as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i

# 載入模型（推薦 fp16）
model = ct.models.MLModel("ckip_ws_fp16.mlpackage")

# Tokenize（BERT 單字切分）
text = "台積電今天股價上漲三十元"
ids = [101] + [vocab.get(ch, 100) for ch in text] + [102]  # 101=[CLS], 102=[SEP], 100=[UNK]
input_ids = np.array([ids], dtype=np.int32)
attention_mask = np.ones_like(input_ids, dtype=np.int32)

# 推論
out = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
preds = np.argmax(out["logits"], axis=-1)[0]

# 解碼斷詞結果（B=0: 詞首, I=1: 詞中）
words, cur = [], ""
for i, ch in enumerate(text):
    p = preds[i + 1]  # +1 跳過 [CLS]
    if p == 0 and cur:
        words.append(cur)
        cur = ch
    else:
        cur += ch
if cur:
    words.append(cur)

print(words)
# ['台積電', '今天', '股價', '上漲', '三十', '元']
```

### 5. 執行詞性標注（Python）

```python
import json

pos_model = ct.models.MLModel("ckip_pos_fp16.mlpackage")
out = pos_model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
preds = np.argmax(out["logits"], axis=-1)[0]

# POS id2label 對照表（從 MLX config.json 取得，或用以下常見標籤）
# 完整對照表見 GitHub repo 的 models/pos/config.json
for i, ch in enumerate(text):
    print(f"{ch} → label_id={preds[i + 1]}")
```

### 6. 在 Swift / iOS 中使用

```swift
import CoreML

// 載入模型
let config = MLModelConfiguration()
config.computeUnits = .all  // 使用 ANE + GPU + CPU
let model = try MLModel(contentsOf: modelURL, configuration: config)

// 準備輸入
let inputIds = try MLMultiArray(shape: [1, seqLen as NSNumber], dataType: .int32)
let attentionMask = try MLMultiArray(shape: [1, seqLen as NSNumber], dataType: .int32)

// 填入 token IDs（[CLS] + 單字 IDs + [SEP]）
for (i, id) in tokenIds.enumerated() {
    inputIds[i] = NSNumber(value: id)
    attentionMask[i] = 1
}

// 推論
let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_ids": inputIds,
    "attention_mask": attentionMask
])
let output = try model.prediction(from: input)
let logits = output.featureValue(for: "logits")!.multiArrayValue!

// 取 argmax 得到預測標籤
```

## 速度

測試環境：Apple M4 Max / 128GB / macOS 26.3.1
測試資料：維基百科「臺灣」條目，36,245 字，10 runs median

| Framework | fp32 | fp16 |
|-----------|-----:|-----:|
| **CoreML** | 2,879 ms | **2,352 ms** ⚡ |
| MLX | 2,869 ms | 3,092 ms |
| HF Transformers (MPS) | 3,532 ms | 3,096 ms |
| CKIP 官方 (MPS) | 14,926 ms | 11,850 ms |

## 跨框架驗證

CoreML fp32 與 MLX fp32、HF Transformers fp32 的 WS/POS/NER 輸出**完全一致**，確認轉換正確。

## 授權

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)，依循原始 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 授權。

## 致謝

- [CKIP Lab, 中央研究院資訊科學研究所](https://ckip.iis.sinica.edu.tw/) — 原始模型
- [Apple CoreML](https://developer.apple.com/documentation/coreml) — 推論框架
