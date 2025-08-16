## Task 3 – Prompt / Caption Embeddings (CLIP Wrapper)

We now transform cleaned text into dense vectors suitable for conditioning generative models. This layer abstracts away tokenisation + forward pass over a CLIP text encoder while keeping an interface simple enough to swap models (e.g., BERT, T5) later.

### Core File
* `embedding_module.py` – `TextEmbedder` loads a CLIP tokenizer + text model and exposes `encode(texts)` returning `(embeddings, attention_mask)`.

### Usage
```python
from task3.embedding_module import TextEmbedder
embedder = TextEmbedder(model_name="openai/clip-vit-base-patch32")
emb, mask = embedder.encode(["a red square", "a blue circle"])
print(emb.shape)  # (batch, seq_len, hidden)
```

### Design Choices
* Returns *last hidden state* instead of a pooled CLS so downstream components can choose pooling strategy (mean, attention, cross‑attn injection).
* Keeps device management simple; moves model to CUDA automatically if available.

### When To Pool?
* GAN label embedding: often mean or first token is enough.
* Cross‑attention (diffusion): keep full sequence.

### Performance Tips
* Batch prompts to reduce tokenizer overhead.
* Cache frequent, static conditioning prompts (e.g., digits 0‑9) once.

### Extensions
* Add optional projection head to a fixed dimension used by the GAN.
* Integrate half precision (fp16) for large batches.
* Support multi‑lingual models for non‑English captions.

### Failure Modes
| Issue | Cause | Mitigation |
|-------|-------|------------|
| OOM on GPU | Long batch × seq len | Reduce batch size; enable fp16 |
| Slow first call | Model weights download | Pre‑download in build image |
