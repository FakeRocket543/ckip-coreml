# CKIP BERT CoreML — 繁體中文 NLP for iOS/macOS

CKIP BERT-base 的 CoreML 版本，可在 iOS/macOS 上透過 Apple Neural Engine (ANE) 執行。

從 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 的 HuggingFace 權重轉換而來。搭配 [ckip-mlx](https://github.com/FakeRocket543/ckip-mlx) 使用。

## 功能

- **WS** — 中文斷詞 (Word Segmentation)
- **POS** — 詞性標注 (Part-of-Speech Tagging, 60 類)
- **NER** — 命名實體辨識 (Named Entity Recognition, 73 類 BIOES)
- 動態序列長度 1–512（解決 CoreML 固定長度限制）
- 支援 ANE 加速

## 模型版本

| 版本 | 單模型大小 | 三模型全載 | 適合裝置 | 精度 (vs fp32) |
|------|--------:|--------:|---------|--------------|
| fp32 | 388 MB | 1.2 GB | 16GB+ Mac | baseline (與 MLX fp32 100% 一致) |
| fp16 | 194 MB | 582 MB | 8GB+ Mac / iPhone 15 Pro+ | WS 100% / POS 99.97% / NER 99.99% |
| q8 | 98 MB | 294 MB | 6GB+ iPhone | WS 99.96% / POS 98.83% / NER 99.76% |

推薦：fp16（速度最快、精度幾乎無損）

## 下載模型

模型權重託管於 HuggingFace：[FakeRockert543/ckip-coreml](https://huggingface.co/FakeRockert543/ckip-coreml)

```bash
pip install huggingface_hub

# 下載全部 .mlpackage
huggingface-cli download FakeRockert543/ckip-coreml --local-dir .
```

## 安裝

```bash
# 需要 Python 3.13（coremltools 9.0 不支援 3.14）
python3.13 -m venv .venv && source .venv/bin/activate
pip install coremltools torch safetensors
```

## 從 MLX 權重轉換

需要先有 MLX 版本的權重（見 [ckip-mlx](https://github.com/FakeRocket543/ckip-mlx)）：

```bash
# 轉換全部 9 個模型（3 任務 × 3 精度）
python convert_all_variants.py
```

## 在 Python 中使用

```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel("ckip_ws_fp16.mlpackage")

text = "台積電今天股價上漲三十元"
# tokenize（單字切分，加 [CLS]=101, [SEP]=102）
input_ids = np.array([[101] + [vocab[ch] for ch in text] + [102]])
attention_mask = np.ones_like(input_ids)

out = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
logits = out["logits"]
preds = np.argmax(logits, axis=-1)[0]
```

## 在 Swift/iOS 中使用

```swift
let model = try MLModel(contentsOf: modelURL)
let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_ids": MLMultiArray(inputIds),
    "attention_mask": MLMultiArray(attentionMask)
])
let output = try model.prediction(from: input)
```

## Benchmark

### 測試環境

Apple M4 Max / 128GB / macOS 26.3.1

### 速度比較 (WS+POS+NER, 36,245 字, 10 runs median)

| Framework | fp32 | fp16 |
|-----------|-----:|-----:|
| **CoreML** | 2,879 ms | **2,352 ms** ⚡ |
| MLX | 2,869 ms | 3,092 ms |
| HF Transformers (MPS) | 3,532 ms | 3,096 ms |
| CKIP 官方 (MPS) | 14,926 ms | 11,850 ms |

CoreML fp16 是所有框架中最快的，比 CKIP 官方快 6.3 倍。

## 檔案結構

```
├── convert_all_variants.py  # MLX → CoreML 轉換（含 BERT 實作）
├── benchmark_variants.py    # CoreML benchmark
├── ckip_{ws,pos,ner}_fp32.mlpackage
├── ckip_{ws,pos,ner}_fp16.mlpackage
└── ckip_{ws,pos,ner}_q8.mlpackage
```

## 授權

[GPL-3.0 License](LICENSE)，依循原始 CKIP Transformers 授權。

原始模型權重來自 [ckiplab](https://github.com/ckiplab/ckip-transformers)（中央研究院資訊科學研究所）。
