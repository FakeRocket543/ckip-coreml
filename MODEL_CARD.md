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

從 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 轉換，經由 [ckip-mlx](https://huggingface.co/FakeRockert543/ckip-mlx) 中繼。

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

## 速度

測試環境：Apple M4 Max / 128GB / macOS 26.3.1
測試資料：維基百科「臺灣」條目，36,245 字

| Framework | fp32 | fp16 |
|-----------|-----:|-----:|
| **CoreML** | 2,879 ms | **2,352 ms** ⚡ |
| MLX | 2,869 ms | 3,092 ms |
| HF Transformers (MPS) | 3,532 ms | 3,096 ms |
| CKIP 官方 (MPS) | 14,926 ms | 11,850 ms |

CoreML fp16 是所有框架中最快的，比 CKIP 官方快 **6.3 倍**。

## 使用方式

### Python

```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel("ckip_ws_fp16.mlpackage")

text = "台積電今天股價上漲三十元"
input_ids = np.array([[101] + [vocab[ch] for ch in text] + [102]])
attention_mask = np.ones_like(input_ids)

out = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
preds = np.argmax(out["logits"], axis=-1)[0]
```

### Swift / iOS

```swift
let model = try MLModel(contentsOf: modelURL)
let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_ids": MLMultiArray(inputIds),
    "attention_mask": MLMultiArray(attentionMask)
])
let output = try model.prediction(from: input)
```

## 量化精度詳細測試

以維基百科「臺灣」條目 36,245 字測試，與 fp32 逐 token 比對（共 36,389 tokens）：

### fp16
- WS: 1 token 不同 (100.00%)
- POS: 11 tokens 不同 (99.97%)
- NER: 3 tokens 不同 (99.99%)

### q8
- WS: 13 tokens 不同 (99.96%)
- POS: 425 tokens 不同 (98.83%)
- NER: 89 tokens 不同 (99.76%)

## 跨框架驗證

CoreML fp32 與 MLX fp32、HF Transformers fp32 的 WS/POS/NER 輸出**完全一致**，確認轉換正確。

## 相關專案

- [FakeRockert543/ckip-mlx](https://huggingface.co/FakeRockert543/ckip-mlx) — MLX 版本（桌面推薦）
- [FakeRocket543/ckip-coreml](https://github.com/FakeRocket543/ckip-coreml) — 原始碼與轉換腳本

## 授權

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)，依循原始 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 授權。

## 致謝

- [CKIP Lab, 中央研究院資訊科學研究所](https://ckip.iis.sinica.edu.tw/) — 原始模型
- [Apple CoreML](https://developer.apple.com/documentation/coreml) — 推論框架
