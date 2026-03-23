"""Benchmark CoreML vs MLX vs PyTorch — correctness + speed."""

import coremltools as ct
import numpy as np
import torch
import json, os, time, statistics, re

# ── Load text ──
with open("../ckip_mlx/wiki_taiwan.txt") as f:
    raw = f.read()
text = re.sub(r'\s+', '', raw)
text = ''.join(ch for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or ch in '，。、；：！？「」『』（）—…──《》〈〉' or '\uff00' <= ch <= '\uffef')

MAX_SEQ = 510
N_RUNS = 10

# ── Tokenizer ──
vocab = {}
with open("../ckip_mlx/models/ws/vocab.txt") as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i
unk = vocab["[UNK]"]; cls = vocab["[CLS]"]; sep = vocab["[SEP]"]

def make_chunks(text):
    chunks = []
    for s in range(0, len(text), MAX_SEQ):
        c = text[s:s+MAX_SEQ]
        ids = [cls] + [vocab.get(ch, unk) for ch in c] + [sep]
        chunks.append((ids, c))
    return chunks

chunks = make_chunks(text)
print(f"文字: {len(text)} 字, {len(chunks)} chunks")

# ── Load CoreML models ──
print("Loading CoreML models...")
cm_models = {}
for task in ["ws", "pos", "ner"]:
    cm_models[task] = ct.models.MLModel(f"ckip_{task}.mlpackage")

def cm_infer(task, ids):
    ids_np = np.array([ids], dtype=np.int32)
    mask_np = np.ones((1, len(ids)), dtype=np.int32)
    out = cm_models[task].predict({"input_ids": ids_np, "attention_mask": mask_np})
    logits = list(out.values())[0]
    return np.argmax(logits, axis=-1).flatten().tolist()

# ── Load MLX baseline preds ──
print("Loading MLX baseline...")
mlx_result = json.load(open("../ckip_mlx/result_mlx_fp32.json"))

# ── Run CoreML on all chunks, get raw preds ──
print("Running CoreML inference...")
all_cm_preds = {"ws": [], "pos": [], "ner": []}
for ids, ct_text in chunks:
    for task in ["ws", "pos", "ner"]:
        preds = cm_infer(task, ids)
        all_cm_preds[task].extend(preds)

# ── Compare token-level preds with MLX ──
# Load MLX raw preds for comparison
import sys
sys.path.insert(0, "../ckip_mlx")
import mlx.core as mx
import mlx.nn as mnn
from bert_mlx import BertForTokenClassification as MLXBert

def load_mlx(task):
    d = f"../ckip_mlx/models/{task}"
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    cfg["num_labels"] = len(cfg.get("id2label", {})) or cfg.get("num_labels", 2)
    m = MLXBert(cfg)
    m.load_weights(os.path.join(d, "weights.safetensors"))
    mx.eval(m.parameters())
    return m

print("Comparing token-level preds (CoreML vs MLX fp32)...")
for task in ["ws", "pos", "ner"]:
    mlx_model = load_mlx(task)
    mlx_preds = []
    for ids, ct_text in chunks:
        o = mlx_model(mx.array([ids]), attention_mask=mx.array([[1]*len(ids)]))
        mx.eval(o)
        mlx_preds.extend(mx.argmax(o, axis=-1).tolist()[0])
    del mlx_model

    total = len(mlx_preds)
    diff = sum(1 for a, b in zip(mlx_preds, all_cm_preds[task]) if a != b)
    print(f"  [{task.upper()}] {total-diff}/{total} 一致 ({(total-diff)/total*100:.2f}%)")

# ── Speed benchmark ──
print(f"\nSpeed benchmark ({N_RUNS} runs)...")

# CoreML speed
for _ in range(2):  # warmup
    for ids, _ in chunks[:5]:
        for t in cm_models: cm_infer(t, ids)

cm_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    for ids, _ in chunks:
        for t in cm_models: cm_infer(t, ids)
    cm_times.append((time.perf_counter() - t0) * 1000)

# MLX speed
mlx_models = {t: load_mlx(t) for t in ["ws","pos","ner"]}
for _ in range(2):
    for ids, _ in chunks[:5]:
        for t in mlx_models:
            o = mlx_models[t](mx.array([ids]), attention_mask=mx.array([[1]*len(ids)]))
            mx.eval(o)

mlx_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    for ids, _ in chunks:
        for t in mlx_models:
            o = mlx_models[t](mx.array([ids]), attention_mask=mx.array([[1]*len(ids)]))
            mx.eval(o)
    mlx_times.append((time.perf_counter() - t0) * 1000)

cm_med = statistics.median(cm_times)
mlx_med = statistics.median(mlx_times)

print(f"\n{'='*50}")
print(f"速度 (WS+POS+NER, {len(text)} 字, {N_RUNS} runs median)")
print(f"{'='*50}")
print(f"  CoreML:  {cm_med:.0f} ms")
print(f"  MLX:     {mlx_med:.0f} ms")
print(f"  比率:    CoreML/MLX = {cm_med/mlx_med:.2f}x")
