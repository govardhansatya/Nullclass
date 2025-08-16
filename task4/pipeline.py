"""Integrated text-to-image mini pipeline combining:
1. Text preprocessing (task1)
2. Text embedding (task3)
3. Conditional GAN with attention (task5 extension)

This script shows how components interoperate end-to-end on a small demo.
"""
from pathlib import Path
import torch
from task1.text import BasicTextProcessor
from task3.embedding_module import TextEmbedder
from task4.gans import AttnGenerator, LATENT_DIM, N_CLASSES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAMPLE_TEXTS = [
    "A handwritten digit zero.",
    "A handwritten digit one.",
    "A handwritten digit two.",
]

def preprocess_and_embed(texts):
    processor = BasicTextProcessor()
    cleaned = [processor.clean_text(t) for t in texts]
    embedder = TextEmbedder()
    hidden, mask = embedder.encode(cleaned)
    # Use CLS token (index 0) as aggregated embedding representation
    cls_embeddings = hidden[:, 0, :]
    return cls_embeddings, cleaned

@torch.no_grad()
def generate_from_labels(generator: torch.nn.Module, labels):
    z = torch.randn(len(labels), LATENT_DIM, device=DEVICE)
    imgs = generator(z, labels.to(DEVICE))
    return imgs


def main():
    # 1. Text -> embeddings (not directly used by current GAN but placeholder for future fusion)
    cls_embs, cleaned = preprocess_and_embed(SAMPLE_TEXTS)
    print('Cleaned texts:', cleaned)
    print('Embedding tensor:', cls_embs.shape)

    # 2. Load trained generator (attention version)
    gen_path = Path('cgan_outputs/models/generator.pt')
    G = AttnGenerator().to(DEVICE)
    if gen_path.exists():
        G.load_state_dict(torch.load(gen_path, map_location=DEVICE))
        print('Loaded trained generator weights.')
    else:
        print('WARNING: Trained generator weights not found, using random initialized model.')
    G.eval()

    # 3. Generate a grid for labels 0..9
    labels = torch.arange(0, N_CLASSES)
    imgs = generate_from_labels(G, labels)
    print('Generated batch shape:', imgs.shape)

    # 4. Save sample grid
    from torchvision.utils import make_grid, save_image
    grid = make_grid(imgs * 0.5 + 0.5, nrow=5)
    Path('pipeline_outputs').mkdir(exist_ok=True)
    save_image(grid, 'pipeline_outputs/sample_digits.png')
    print('Saved pipeline_outputs/sample_digits.png')

if __name__ == '__main__':
    main()
