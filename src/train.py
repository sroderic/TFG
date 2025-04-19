import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
import time

from HAM10000Dataset import HAM10000Dataset
from unet import UNet
import utils

# Configuració
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carreguem els datasets i transforms
train_dataset = HAM10000Dataset(
	utils.df_train, utils.images_folder, utils.masks_folder,
	image_transform=utils.image_transform,
	mask_transform=utils.mask_transform
)
val_dataset = HAM10000Dataset(
	utils.df_val, utils.images_folder, utils.masks_folder,
	image_transform=utils.image_transform,
	mask_transform=utils.mask_transform
)

train_loader = DataLoader(train_dataset,
						  batch_size=BATCH_SIZE,
						  shuffle=True,
						  num_workers=2,
						  pin_memory=True)
val_loader = DataLoader(val_dataset,
						 batch_size=BATCH_SIZE,
						  shuffle=True,
						  num_workers=2,
						  pin_memory=True)

# Inicialitzem model, pèrdua i optimitzador
NUM_CLASSES = len(utils.class_to_int)
model = UNet(in_channels=3, num_classes=NUM_CLASSES, padding=0).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Guardem el millor model
best_val_loss = float('inf')
model_dir = Path("checkpoints")
model_dir.mkdir(exist_ok=True)

# Entrenament
for epoch in range(NUM_EPOCHS):
	start_epoch = time.time()
	loading_time = 0
	start_load_time = start_epoch

	model.train()
	train_loss = 0.0

	for images, masks in tqdm(train_loader, desc="Training", leave=False):

		images = images.to(DEVICE)
		masks = masks.squeeze(1).long().to(DEVICE)  # (B, H, W)
		end_load_time = time.time() - start_load_time
		loading_time += end_load_time
		optimizer.zero_grad()
		outputs = model(images)  # (B, C, H, W)

		loss = criterion(outputs, masks)
		loss.backward()
		optimizer.step()

		train_loss += loss.item() * images.size(0)
		start_load_time = time.time()

	avg_train_loss = train_loss / len(train_loader.dataset)
	end_epoch = time.time() - start_epoch
	print(f"Time total epoch: {end_epoch}")
	print(f"Time loading dataset: {loading_time}")
	print(f"Time training: {end_epoch - loading_time}")
	exit()
	# VALIDACIÓ
	model.eval()
	val_loss = 0.0
	with torch.no_grad():
		for images, masks in tqdm(val_loader, desc="Validation", leave=False):
			images = images.to(DEVICE)
			masks = masks.squeeze(1).long().to(DEVICE)

			outputs = model(images)
			loss = criterion(outputs, masks)
			val_loss += loss.item() * images.size(0)

	avg_val_loss = val_loss / len(val_loader.dataset)

	print(f"📘 Epoch [{epoch+1}/{NUM_EPOCHS}]")
	print(f"   🟢 Train Loss: {avg_train_loss:.4f}")
	print(f"   🔵 Val   Loss: {avg_val_loss:.4f}")

	# Guardem si és el millor
	if avg_val_loss < best_val_loss:
		best_val_loss = avg_val_loss
		# torch.save(model.state_dict(), model_dir / "best_model.pth")
		print("💾 Millor model actualitzat!")

print("🏁 Entrenament complet.")