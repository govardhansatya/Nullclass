## Task 1 – Text Cleaning & Basic NLP Prep

This first step lays the groundwork for everything that follows. We take raw text (news, captions, prompts) and turn it into something consistent, lower‑noise, and ready for tokenisation or embedding.

### What It Does
* Normalises whitespace and punctuation.
* Lowercases and strips out obvious junk tokens.
* Splits into sentences and words (NLTK punkt if available; falls back gracefully if not).
* Provides simple corpus stats (average length, vocabulary size estimation, etc.).

### Key File
* `text.py` – Contains `BasicTextProcessor` with `clean_text`, `tokenise_sentences`, `tokenise_words`, and `preprocess` helpers.

### Quick Start
```python
from task1.text import BasicTextProcessor
processor = BasicTextProcessor()
clean_lines = processor.preprocess(["A Sample   LINE!!  With   spacing..."])
print(clean_lines)
```

### Design Notes
* Keeps dependencies light so it can run in restricted training environments.
* Avoids aggressive stemming / lemmatisation to preserve semantics for downstream embeddings.

### Common Extensions
* Add language detection + route to language‑specific normalisers.
* Integrate emoji / unicode normalisation.
* Plug in a profanity / PII redaction pass before logging.

### Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Empty output list | All lines filtered as noise | Pass `keep_empty=True` or inspect cleaning regex |
| Slow first run | NLTK downloading punkt | Pre-download in build step |

### Why It Matters
Clean input reduces variance for the GAN label conditioning and improves stability for CLIP / diffusion prompt embeddings later on. Garbage in → muddier gradients later.
