import os
import multiprocessing
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict

# Optional: sklearn metrics for on-the-fly conditional accuracy evaluation
try:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False

# ───────────────────────────────────────────────────────────────────────────────
# 1. Model Definitions
# ───────────────────────────────────────────────────────────────────────────────
LATENT_DIM = 100
N_CLASSES  = 10
EMBED_DIM  = 50
IMG_SIZE   = 28
CHANNELS   = 1

# -----------------------------------------------------------------------------
# Self-Attention Block (SAGAN style simplified for single-head)
# -----------------------------------------------------------------------------
class SelfAttention2d(nn.Module):
    """A light-weight self-attention layer for 2D feature maps."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.f = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.g = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.h = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.v = nn.Conv2d(in_channels // 2, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        f = self.f(x).view(B, -1, H * W)            # B, C', N
        g = self.g(x).view(B, -1, H * W)            # B, C', N
        h = self.h(x).view(B, -1, H * W)            # B, C'', N
        beta = torch.softmax(torch.bmm(f.transpose(1, 2), g), dim=-1)  # B, N, N
        o = torch.bmm(h, beta.transpose(1, 2))       # B, C'', N
        o = o.view(B, -1, H, W)
        o = self.v(o)
        return self.gamma * o + x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, CHANNELS * IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z, labels):
        y = self.label_emb(labels)
        x = torch.cat([z, y], dim=1)
        img = self.model(x)
        return img.view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)


class AttnGenerator(nn.Module):
    """Convolutional generator with a self-attention block at 14x14 resolution."""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.project = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Self-attention inserted here
            SelfAttention2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, CHANNELS, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y = self.label_emb(labels)
        h = torch.cat([z, y], dim=1)
        h = self.project(h)
        h = h.view(-1, 128, 7, 7)
        img = self.net(h)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.model = nn.Sequential(
            nn.Linear(CHANNELS * IMG_SIZE * IMG_SIZE + EMBED_DIM, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        y = self.label_emb(labels)
        x = img.view(img.size(0), -1)
        x = torch.cat([x, y], dim=1)
        return self.model(x)

def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ───────────────────────────────────────────────────────────────────────────────
# 2. Training Function
# ───────────────────────────────────────────────────────────────────────────────
def train(use_attention: bool = True,
        epochs: int = 30,
        save_dir: str = 'cgan_outputs',
        evaluate: bool = True) -> Dict[str, list]:
    # Configuration
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    LR         = 0.0002
    BETA1      = 0.5
    EPOCHS     = epochs
    OUTPUT_DIR = save_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)

    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mnist = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        mnist,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize models
    G = (AttnGenerator() if use_attention else Generator()).to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    adversarial_loss = nn.BCELoss().to(DEVICE)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    def sample_image(n_row, epoch):
        z = torch.randn(n_row**2, LATENT_DIM, device=DEVICE)
        labels = torch.arange(n_row).repeat_interleave(n_row).to(DEVICE)
        gen_imgs = G(z, labels)
        gen_imgs = gen_imgs * 0.5 + 0.5  # denormalize
        grid = torchvision.utils.make_grid(gen_imgs, nrow=n_row, normalize=False)
        torchvision.utils.save_image(grid, f"{OUTPUT_DIR}/epoch_{epoch}.png")

    history = {"g_loss": [], "d_loss": []}

    # Optional classifier for conditional accuracy evaluation
    classifier = None
    if evaluate and _HAVE_SK:
        classifier = _build_or_load_mnist_classifier(DEVICE)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        for imgs, labels in dataloader:
            batch_size = imgs.size(0)
            real_imgs  = imgs.to(DEVICE)
            labels     = labels.to(DEVICE)

            valid = torch.ones(batch_size, 1, device=DEVICE)
            fake  = torch.zeros(batch_size, 1, device=DEVICE)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred = D(real_imgs, labels)
            loss_real = adversarial_loss(real_pred, valid)
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = G(z, labels)
            fake_pred = D(gen_imgs.detach(), labels)
            loss_fake = adversarial_loss(fake_pred, fake)
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            gen_pred = D(gen_imgs, labels)
            g_loss = adversarial_loss(gen_pred, valid)
            g_loss.backward()
            optimizer_G.step()

        print(f"[Epoch {epoch}/{EPOCHS}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
        sample_image(n_row=10, epoch=epoch)
        history["g_loss"].append(g_loss.item())
        history["d_loss"].append(d_loss.item())

        # Conditional accuracy snapshot every 5 epochs
        if evaluate and _HAVE_SK and epoch % 5 == 0:
            cond_metrics = conditional_accuracy(G, classifier, DEVICE)
            print("  > Conditional Acc: {acc:.2f}%".format(acc=cond_metrics['accuracy'] * 100))

    # Save final model weights
    torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, 'models', 'generator.pt'))
    torch.save(D.state_dict(), os.path.join(OUTPUT_DIR, 'models', 'discriminator.pt'))

    print("Training complete. Samples saved to", OUTPUT_DIR)
    return history


# -----------------------------------------------------------------------------
# Evaluation Helpers
# -----------------------------------------------------------------------------
class _SimpleMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def _build_or_load_mnist_classifier(device: torch.device) -> nn.Module:
    """Train a quick classifier (1 epoch) for conditional accuracy metrics."""
    clf_path = 'cgan_outputs/models/mnist_classifier.pt'
    model = _SimpleMNISTClassifier().to(device)
    if os.path.exists(clf_path):
        model.load_state_dict(torch.load(clf_path, map_location=device))
        model.eval()
        return model
    # Train quickly
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for batch, (x, y) in enumerate(loader):
        if batch > 400:  # ~50k samples subset for speed
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    os.makedirs('cgan_outputs/models', exist_ok=True)
    torch.save(model.state_dict(), clf_path)
    return model


def conditional_accuracy(G: nn.Module, classifier: nn.Module, device: torch.device) -> Dict[str, float]:
    """Generate labeled samples and compute confusion matrix & metrics."""
    G.eval()
    with torch.no_grad():
        labels = torch.arange(0, 10, device=device).repeat_interleave(100)  # 100 per class
        z = torch.randn(len(labels), LATENT_DIM, device=device)
        imgs = G(z, labels)
        imgs_norm = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
        logits = classifier(imgs_norm)
        preds = torch.argmax(logits, dim=1)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = (y_true == y_pred).mean()
    # Save confusion matrix for inspection
    import numpy as np
    os.makedirs('cgan_outputs/metrics', exist_ok=True)
    np.savetxt('cgan_outputs/metrics/confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def _parse_args():
    p = argparse.ArgumentParser(description='Train Conditional (Attention) GAN on MNIST')
    p.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', '30')))
    p.add_argument('--no-attn', action='store_true', help='Disable attention generator')
    p.add_argument('--no-eval', action='store_true', help='Disable conditional accuracy evaluation')
    p.add_argument('--out', type=str, default='cgan_outputs')
    return p.parse_args()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows support for spawning
    args = _parse_args()
    train(use_attention=not args.no_attn, epochs=args.epochs, evaluate=not args.no_eval, save_dir=args.out)