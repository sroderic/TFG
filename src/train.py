from HAM10000Dataset import HAM10000Dataset
from torch.utils.data import DataLoader
from unet2 import UNet
from torch import nn
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
import torch

def train(config):
	# Carreguem els datasets i transforms
	train_dataset = HAM10000Dataset(
		config.df_train,
		config.data_folder,
		config.image_size,
		config.preload,
		config.padding
	)
	val_dataset = HAM10000Dataset(
		config.df_val,
		config.data_folder,
		config.image_size,
		config.preload,
		config.padding
	)
	
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch,
		shuffle=True,
		num_workers=config.num_workers,
		# pin_memory=True
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=config.batch,
		shuffle=True,
		num_workers=config.num_workers,
		# pin_memory=True
	)

	# Inicialitzem model, pèrdua i optimitzador
	model = UNet(
		in_channels=3,
		num_classes=config.num_classes,
		padding= 1 if config.padding else 0
	).to(config.device)

	# Loss function i optimitzador
	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=config.lr)

	# Guardem el millor model
	best_val_loss = float('inf')
	model_dir = Path("checkpoints")
	model_dir.mkdir(exist_ok=True)

	# Entrenament
	for epoch in range(config.epochs):

		model.train()
		train_loss = 0.0

		for images, masks in tqdm(train_loader, desc="Training"):

			images = images.to(config.device)
			masks = masks.squeeze(1).long().to(config.device)  # (B, H, W)
			optimizer.zero_grad()
			outputs = model(images)  # (B, C, H, W)

			loss = criterion(outputs, masks)
			loss.backward()
			optimizer.step()

			train_loss += loss.item() * images.size(0)

		avg_train_loss = train_loss / len(train_loader.dataset)

		# VALIDACIÓ
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for images, masks in tqdm(val_loader, desc="Validation"):
				images = images.to(config.device)
				masks = masks.squeeze(1).long().to(config.device)

				outputs = model(images)
				loss = criterion(outputs, masks)
				val_loss += loss.item() * images.size(0)

		avg_val_loss = val_loss / len(val_loader.dataset)

		print(f"📘 Epoch [{epoch+1}/{config.epochs}]")
		print(f"   🟢 Train Loss: {avg_train_loss:.4f}")
		print(f"   🔵 Val   Loss: {avg_val_loss:.4f}")

		# Guardem si és el millor
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			# torch.save(model.state_dict(), model_dir / "best_model.pth")
			print("💾 Millor model actualitzat!")

	print("🏁 Entrenament complet.")