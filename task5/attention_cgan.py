"""Attention Conditional GAN on Synthetic Shapes (Task 5)

Extends Task 4 by adding a self-attention block in the generator and discriminator.
Generates 32x32 RGB images of shapes (square, circle, triangle).
"""
import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from task4.shapes_dataset import ShapesDataset, SHAPE_LABELS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
N_CLASSES = len(SHAPE_LABELS)
IMG_SIZE = 32
CHANNELS = 3

class SelfAttention(nn.Module):
	def __init__(self, in_ch):
		super().__init__()
		self.query = nn.Conv2d(in_ch, in_ch//8, 1)
		self.key   = nn.Conv2d(in_ch, in_ch//8, 1)
		self.value = nn.Conv2d(in_ch, in_ch, 1)
		self.gamma = nn.Parameter(torch.zeros(1))
	def forward(self, x):
		B,C,H,W = x.shape
		q = self.query(x).view(B,-1,H*W).permute(0,2,1)
		k = self.key(x).view(B,-1,H*W)
		attn = torch.softmax(torch.bmm(q,k), dim=-1)
		v = self.value(x).view(B,C,H*W)
		o = torch.bmm(v, attn.permute(0,2,1)).view(B,C,H,W)
		return self.gamma * o + x

class Gen(nn.Module):
	def __init__(self):
		super().__init__()
		self.embed = nn.Embedding(N_CLASSES, 32)
		self.fc = nn.Linear(LATENT_DIM+32, 256*4*4)
		self.net = nn.Sequential(
			nn.BatchNorm2d(256),
			nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True),
			SelfAttention(128),
			nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2,True),
			nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2,True),
			nn.Conv2d(32, CHANNELS, 3,1,1), nn.Tanh()
		)
	def forward(self, z, labels):
		y = self.embed(labels)
		h = torch.cat([z,y],1)
		h = self.fc(h).view(-1,256,4,4)
		return self.net(h)

class Disc(nn.Module):
	def __init__(self):
		super().__init__()
		self.embed = nn.Embedding(N_CLASSES, 32)
		self.net = nn.Sequential(
			nn.Conv2d(CHANNELS+1,64,4,2,1), nn.LeakyReLU(0.2,True),
			SelfAttention(64),
			nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True),
			nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2,True),
			nn.Conv2d(256,1,4,1,0)
		)
	def forward(self, x, labels):
		B = x.size(0)
		y = self.embed(labels)
		y_img = y.view(B,1,1,32).repeat(1,1,IMG_SIZE,1)[:,:,:,:IMG_SIZE]
		x_in = torch.cat([x,y_img],1)
		return self.net(x_in).view(-1,1)

def train(epochs=50, out='task5_shapes_outputs'):
	os.makedirs(out, exist_ok=True)
	ds = ShapesDataset(length=3000)
	dl = DataLoader(ds, batch_size=64, shuffle=True)
	G, D = Gen().to(DEVICE), Disc().to(DEVICE)
	optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
	optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
	loss_fn = nn.BCEWithLogitsLoss()
	for epoch in range(1, epochs+1):
		for real, labels in dl:
			real, labels = real.to(DEVICE), labels.to(DEVICE)
			B = real.size(0)
			valid = torch.ones(B,1,device=DEVICE); fake = torch.zeros(B,1,device=DEVICE)
			# D
			optD.zero_grad()
			z = torch.randn(B, LATENT_DIM, device=DEVICE)
			gen = G(z, labels)
			d_real = loss_fn(D(real, labels), valid)
			d_fake = loss_fn(D(gen.detach(), labels), fake)
			d_loss = (d_real + d_fake)/2; d_loss.backward(); optD.step()
			# G
			optG.zero_grad()
			g_loss = loss_fn(D(gen, labels), valid); g_loss.backward(); optG.step()
		if epoch % 10 == 0:
			from torchvision.utils import save_image, make_grid
			with torch.no_grad():
				test_labels = torch.arange(0,N_CLASSES, device=DEVICE)
				z = torch.randn(N_CLASSES, LATENT_DIM, device=DEVICE)
				imgs = G(z, test_labels)*0.5+0.5
				save_image(imgs, f"{out}/epoch_{epoch}.png", nrow=N_CLASSES)
		print(f"Epoch {epoch}/{epochs} D {d_loss.item():.3f} G {g_loss.item():.3f}")
	torch.save(G.state_dict(), f"{out}/generator.pt")
	torch.save(D.state_dict(), f"{out}/discriminator.pt")
	print('Training complete.')

if __name__ == '__main__':
	ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=50)
	ap.add_argument('--out', type=str, default='task5_shapes_outputs')
	args = ap.parse_args(); train(args.epochs, args.out)
