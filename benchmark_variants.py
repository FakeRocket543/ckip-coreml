"""Benchmark CoreML fp32/fp16/q8 — accuracy + speed."""

import coremltools as ct
import numpy as np
import json, os, re, time, statistics, sys

sys.path.insert(0, "../ckip_mlx")
import mlx.core as mx
from bert_mlx import BertForTokenClassification as MLXBert

with open("../ckip_mlx/wiki_taiwan.txt") as f:
    raw = f.read()
text = re.sub(r'\s+', '', raw)
text = ''.join(ch for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or ch in '，。、；：！？「」『』（）—…──《》〈〉' or '\uff00' <= ch <= '\uffef')

MAX_SEQ, N_RUNS = 510, 10

vocab = {}
with open("../ckip_mlx/models/ws/vocab.txt") as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i
unk, cls_id, sep_id = vocab["[UNK]"], vocab["[CLS]"], vocab["[SEP]"]

chunks = []
for s in range(0, len(text), MAX_SEQ):
    c = text[s:s+MAX_SEQ]
    ids = [cls_id] + [vocab.get(ch, unk) for ch in c] + [sep_id]
    chunks.append(ids)

print(f"文字: {len(text)} 字, {len(chunks)} chunks, {N_RUNS} runs\n")

# ── MLX fp32 baseline preds ──
print("Loading MLX fp32 baseline...")
mlx_preds = {}
for task in ["ws","pos","ner"]:
    d = f"../ckip_mlx/models/{task}"
    with open(os.path.join(d,"config.json")) as f:
        cfg = json.load(f)
    cfg["num_labels"] = len(cfg.get("id2label",{})) or cfg.get("num_labels",2)
    m = MLXBert(cfg)
    m.load_weights(os.path.join(d,"weights.safetensors"))
    mx.eval(m.parameters())
    preds = []
    for ids in chunks:
        o = m(mx.array([ids]), attention_mask=mx.array([[1]*len(ids)]))
        mx.eval(o)
        preds.extend(mx.argmax(o, axis=-1).tolist()[0])
    mlx_preds[task] = preds
    del m

# ── CoreML variants ──
variants = ["fp32", "fp16", "q8"]
results = {}

for v in variants:
    print(f"\n── CoreML {v} ──")
    models = {}
    for task in ["ws","pos","ner"]:
        models[task] = ct.models.MLModel(f"ckip_{task}_{v}.mlpackage")

    def infer(task, ids):
        out = models[task].predict({
            "input_ids": np.array([ids], dtype=np.int32),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int32),
        })
        return np.argmax(list(out.values())[0], axis=-1).flatten().tolist()

    # Accuracy
    for task in ["ws","pos","ner"]:
        preds = []
        for ids in chunks:
            preds.extend(infer(task, ids))
        total = len(mlx_preds[task])
        diff = sum(1 for a,b in zip(mlx_preds[task], preds) if a != b)
        print(f"  [{task.upper()}] {total-diff}/{total} ({(total-diff)/total*100:.2f}%)")

    # Speed warmup
    for _ in range(2):
        for ids in chunks[:5]:
            for t in models: infer(t, ids)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for ids in chunks:
            for t in models: infer(t, ids)
        times.append((time.perf_counter() - t0) * 1000)
    results[v] = statistics.median(times)
    print(f"  速度: {results[v]:.0f} ms")

# ── Summary ──
print(f"\n{'='*60}")
print(f"CoreML 速度比較 ({len(text)} 字, {N_RUNS} runs median)")
print(f"{'='*60}")
for v in variants:
    print(f"  {v:>4}: {results[v]:>8.0f} ms")
