# Task 5 – Attention‑Augmented Conditional GAN (Shapes & Digits)

We refine the conditional generator idea by injecting a lightweight self‑attention block so the model can coordinate distant pixels (useful for crisp edges and shape interiors). Two variants are demonstrated:
* MNIST digits with an attention generator (28×28, grayscale).
* Synthetic coloured shapes (square / circle / triangle) at 32×32 RGB.

### Key Files
| File | Purpose |
|------|---------|
| `attention_cgan.py` | Training loop for shapes with attention in both generator + discriminator |
| `task5_attention_cgan_training.ipynb` | Notebook training attention MNIST variant + loss curves |
| `metrics_task5.ipynb` | FID (vs real MNIST subset) + conditional accuracy + confusion matrix export |

### Train (Shapes)
```bash
python -m task5.attention_cgan --epochs 50 --out task5_shapes_outputs
```

### Evaluate (MNIST Attention Model)
Open the metrics notebook and run all cells, or adapt the code snippet inside to a script. Outputs:
* `task5_samples/` – Generated images.
* `task5_real/` – Real subset cached for FID.
* `task5_metrics.json` – Aggregated FID + accuracy + precision/recall/F1.

### Why Attention Helps Here
Even small images benefit: the model can better ensure global symmetry (e.g., roundness of a circle) rather than relying only on local convolutional receptive fields.

### Practical Tips
* If training diverges, reduce learning rate or temporarily disable attention to verify baseline still works.
* Cache classifier weights to avoid retraining for every metrics run.

### Possible Extensions
* Multi‑head self‑attention or transformer blocks for larger images.
* Fuse CLIP text embeddings (Task 3) as richer conditioning than plain label IDs.
* Replace BCE with Wasserstein + gradient penalty for smoother training dynamics.

### Troubleshooting
| Issue | Cause | Mitigation |
|-------|-------|------------|
| FID not improving | Too few epochs / small batch | Increase epochs; add mild augmentation |
| Blurry circle edges | Attention depth too shallow | Add a second attention block at lower resolution |

### Relation to Later Tasks
Establishes the pattern of leveraging attention for structure, which carries over conceptually into diffusion model cross‑attention and LoRA fine‑tuning.

