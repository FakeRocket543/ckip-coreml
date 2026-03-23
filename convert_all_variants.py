"""Convert all CKIP tasks to CoreML: fp32 / fp16 / q8 variants."""

import torch
import torch.nn as nn
import numpy as np
import json, os
from safetensors import safe_open
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_pos, type_vocab_size=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        tok_type = torch.zeros_like(input_ids)
        return self.LayerNorm(self.word_embeddings(input_ids) + self.position_embeddings(pos_ids) + self.token_type_embeddings(tok_type))

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
    def forward(self, x, mask):
        B, L, _ = x.shape; h, d = self.num_heads, self.head_dim
        q = self.query(x).reshape(B,L,h,d).permute(0,2,1,3)
        k = self.key(x).reshape(B,L,h,d).permute(0,2,1,3)
        v = self.value(x).reshape(B,L,h,d).permute(0,2,1,3)
        scores = torch.matmul(q, k.transpose(-2,-1)) / (d**0.5) + mask
        return torch.matmul(torch.softmax(scores, dim=-1), v).permute(0,2,1,3).reshape(B,L,-1)

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.attention_output_dense = nn.Linear(hidden_size, hidden_size)
        self.attention_output_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
    def forward(self, x, mask):
        attn = self.attention_output_dense(self.attention(x, mask))
        x = self.attention_output_LayerNorm(attn + x)
        h = self.output_dense(torch.nn.functional.gelu(self.intermediate_dense(x)))
        return self.output_LayerNorm(h + x)

class BertForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        hs = config["hidden_size"]
        self.embeddings = BertEmbeddings(config["vocab_size"], hs, config["max_position_embeddings"])
        self.layers = nn.ModuleList([BertLayer(hs, config["num_attention_heads"], config["intermediate_size"]) for _ in range(config["num_hidden_layers"])])
        self.classifier = nn.Linear(hs, config["num_labels"])
    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -1e9
        for layer in self.layers: x = layer(x, mask)
        return self.classifier(x)

def load_model(task_dir):
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    weights = {}
    with safe_open(os.path.join(task_dir, "weights.safetensors"), framework="pt") as f:
        for k in f.keys(): weights[k] = f.get_tensor(k)
    model.load_state_dict(weights, strict=True)
    return model.eval(), config

MLX_DIR = "../ckip_mlx/models"

for task in ["ws", "pos", "ner"]:
    print(f"\n{'='*50}")
    print(f"[{task.upper()}]")
    model, config = load_model(f"{MLX_DIR}/{task}")

    # Trace once
    dummy_ids = torch.randint(0, 21128, (1, 128))
    dummy_mask = torch.ones(1, 128, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_ids, dummy_mask))

    ct_inputs = [
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
    ]

    # fp32
    ml32 = ct.convert(traced, inputs=ct_inputs, minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT32)
    ml32.save(f"ckip_{task}_fp32.mlpackage")

    # fp16
    ml16 = ct.convert(traced, inputs=ct_inputs, minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
    ml16.save(f"ckip_{task}_fp16.mlpackage")

    # q8 — use coremltools post-training quantization on the fp32 model
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config_q = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    ml8 = ct.optimize.coreml.linear_quantize_weights(ml32, config=config_q)
    ml8.save(f"ckip_{task}_q8.mlpackage")

    for v in ["fp32", "fp16", "q8"]:
        p = f"ckip_{task}_{v}.mlpackage"
        sz = sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fns in os.walk(p) for f in fns) / 1024/1024
        print(f"  {v}: {sz:.1f} MB")

print("\nDone!")
