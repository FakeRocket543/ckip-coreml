"""Convert CKIP BERT WS to CoreML via ONNX (avoids unsupported ops)."""

import torch
import numpy as np
import os

print("1. Export to ONNX...")
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ws")
model.eval()

seq_len = 128
dummy_ids = torch.randint(0, 21128, (1, seq_len))
dummy_mask = torch.ones(1, seq_len, dtype=torch.long)

torch.onnx.export(
    model, (dummy_ids, dummy_mask),
    "ckip_ws.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "logits": {1: "seq_len"},
    },
    opset_version=17,
)
print("   → ckip_ws.onnx")

print("2. Convert ONNX → CoreML...")
import coremltools as ct

mlmodel = ct.convert(
    "ckip_ws.onnx",
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
    ],
    minimum_deployment_target=ct.target.macOS15,
)

mlmodel.save("ckip_ws.mlpackage")
print("   → ckip_ws.mlpackage")
print("Done!")
