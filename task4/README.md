## Task 4 – Conditional GAN (MNIST Baseline) + Evaluation

Establishes a clear, fast baseline: a label‑conditioned GAN on MNIST digits, with an *optional* attention generator variant, basic training loop, and quantitative sanity metrics (accuracy, precision, recall, F1 via an auxiliary classifier).

### Components
| File | Purpose |
|------|---------|
| `gans.py` | Defines `Generator`, `AttnGenerator`, `Discriminator`, training loop, conditional accuracy helpers |
| `evaluate_gan.py` | Loads a saved generator, samples a grid, computes metrics |
| `pipeline.py` | Mini end‑to‑end demo tying text → embedding → generation |
| `shapes_dataset.py` | Synthetic geometric shapes dataset (square / circle / triangle) for later tasks |

### Quick Train
```bash
python -m task4.gans --epochs 5            # attention on (default)
python -m task4.gans --epochs 5 --no-attn  # MLP baseline
```

Artifacts land in `cgan_outputs/` (sample grids + model weights + metrics).

### Metrics Logic
1. Generate N labeled samples per digit.
2. Classify with a compact CNN (trained quickly or loaded if cached).
3. Compute confusion matrix + macro precision/recall/F1.

### Why An Auxiliary Classifier?
It’s a cheap proxy for conditional fidelity without needing Inception features. If it can reliably recover the intended digit from synthetic images, the generator is learning label → visual mapping.

### Extending the Baseline
* Add spectral norm to discriminator.
* Swap BCE for hinge loss (often stabilises training).
* Track FID (already supported indirectly in later tasks) for fuller quality signal.

### Common Gotchas
| Symptom | Explanation | Action |
|---------|-------------|--------|
| Mode collapse after many epochs | Learning rate too high / insufficient label mixing | Lower LR; add diffusion noise / augmentations |
| Accuracy plateaus early | Generator capacity insufficient | Enable attention variant or widen channels |

### Shapes Dataset Tie‑In
The shapes dataset creates a bridge toward more semantic prompts (e.g., “red triangle”) by providing a controllable low‑complexity domain before tackling natural images.
