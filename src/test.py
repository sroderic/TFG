import time
import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
	def __init__(self, smooth = 0.0):
		super(DiceLoss, self).__init__()
		self.smooth = smooth
		self.eps = 1e-7

	def forward(self, y_pred, y_true):
		y_pred = F.softmax(y_pred, dim=1)
		bs = y_true.size(0) # batch size
		num_classes = y_pred.size(1)  # Number of classes (C)
		
		y_true = y_true.view(bs, -1) # [N, HxW]
		y_pred = y_pred.view(bs, num_classes, -1) # [N, C, HxW]

		y_true = F.one_hot(y_true, num_classes)  # [N, HxW] -> [N,HxW, C]
		y_true = y_true.permute(0, 2, 1)  #  [N,HxW, C] -> [N, C, H*W]

		output = y_pred
		target = y_true.type_as(y_pred)

		intersection = torch.sum(output * target, dim=(0,2))
		union = torch.sum(output + target, dim=(0,2))
		dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth).clamp_min(self.eps)
		loss = 1.0 - dice_score
		# mask = y_true.sum((0, 2)) > 0
		# loss *= mask.to(loss.dtype)
		return loss.mean()

def dice_coeff(y_pred, y_true, smooth=0.0, eps = 1e-7, return_per_class=False):

	bs = y_true.size(0) # batch size
	num_classes = y_pred.size(1)  # Number of classes (C)
	
	y_pred = F.softmax(y_pred, dim=1)  # [N, C, H, W] Probabilities
	y_pred = torch.argmax(y_pred, dim=1) # [N, H, W] Predicted classes

	
	y_true = y_true.view(bs, -1) # [N, HxW]
	y_pred = y_pred.view(bs, -1) # [N, HxW]
	
	y_true = F.one_hot(y_true, num_classes)  # [N, HxW] -> [N,HxW, C]
	y_true = y_true.permute(0, 2, 1)  #  [N,HxW, C] -> [N, C, H*W]

	y_pred = F.one_hot(y_pred, num_classes)  # [N, HxW] -> [N,HxW, C]
	y_pred = y_pred.permute(0, 2, 1)  #  [N,HxW, C] -> [N, C, H*W]
	
	output = y_pred
	target = y_true
	intersection = torch.sum(output * target, dim=(0,2))
	union = torch.sum((output + target) > 0, dim=(0,2))
	dice_score = (2 * intersection + smooth) / (union + smooth).clamp_min(eps)

	return dice_score.mean()

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
	running_loss = 0.
	for images, masks in tqdm(train_loader, desc=f"Training"):
		images = images.to(device)
		targets = masks.long().to(device)  # (B, H, W)
		optimizer.zero_grad()
		logits = model(images)  # (B, C, H, W)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	return running_loss / len(train_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, checkpoints_folder, experiment):

	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Metrics per epoch
	
	metrics = {
		'train_loss': [],
		'val_loss': [],
		'dice_coeff': []
	}
	
	
	# To save the best model
	best_val_loss = float('inf')
	experiment_folder = checkpoints_folder / f'{experiment}'
	experiment_folder.mkdir(exist_ok=True)

	start_training = time.time()
	for epoch in range(epochs):
		print(f"📘 Epoch [{epoch+1}/{epochs}]")
		# Training
		# model.train()
		# avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
		# metrics['train_loss'].append(avg_train_loss)
		# print(f"   🟢 Train Loss: {avg_train_loss:.4f} -- Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}")
	
		# Validation	
		val_loss = 0.
		val_dice = 0.
		model = model.eval()
		with torch.no_grad():
			for images, masks in tqdm(val_loader, desc=f"Validation"):
				images = images.to(device)
				targets = masks.long().to(device)  # (B, H, W)
		
				# forward
				logits = model(images)  # (B, C, H, W)
				loss = criterion(logits, targets)
		
				val_loss += loss.item()
				dice = dice_coeff(logits, targets) # [B, C]
				val_dice += dice.item()
		
		avg_val_loss = val_loss / len(val_loader)
		avg_val_dice = val_dice / len(val_loader)

		metrics['val_loss'].append(avg_val_loss)
		metrics['dice_coeff'].append(avg_val_dice)

		print(f"   🔵 Val   Loss      : {avg_val_loss:.4f} -- Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}")
		print(f'   🔵 Dice Coefficient: {avg_val_dice:4f}')

				
		# Save best model
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			
			best_model_file = experiment_folder / 'best_model.pth'
			torch.save(model.state_dict(), best_model_file)
			print("💾 Best model saved!")
		print('........................................................................')

	metrics_file = experiment_folder / 'metrics.pt'
	torch.save(metrics, metrics_file)
	print("🏁 Entrenament complet.")

if __name__ == "__main__":
	import argparse
	import torch
	import random
	import numpy as np
	from pathlib import Path
	import os
	import pickle
	from model import UNet
	from HAM10000Dataset import HAM10000Dataset
	from torch.utils.data import DataLoader
	from torch import nn, optim
	

	import torch

	# # Settings
	# batch_size = 16
	# num_classes = 3
	# height = 450
	# width = 600

	# # Random logits (before softmax)
	# pred = torch.randn(batch_size, num_classes, height, width)


	# # Random ground truth labels (each pixel has a class index from 0 to 6)
	# target = torch.randint(0, num_classes, (batch_size, height, width))
	# print(DiceLoss().forward(pred, target))
	# print(dice_coeff(pred, target))
	
	# exit()


	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--colab', action='store_true')
	# parser.add_argument('--mode', type=str, required=True, help='train, test, inference')
	parser.add_argument('--image_size', type=int, nargs=2, default=[428, 572], help='Input image size as two integers')
	parser.add_argument('--padding', action='store_true')
	parser.add_argument('--epochs', type=int, required=True)
	parser.add_argument('--batch', type=int, required=True)
	parser.add_argument('--lr', type=float, required=True)
	parser.add_argument('--loss', type=str, required=True, help='Cross, Dice, Focal')
	parser.add_argument('--optimizer', type=str, default='Adam', help='Adam, RMSprop, SGD')
	parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')	

	args = parser.parse_args()
	seed = 42
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# Data folders
	if args.colab:
		root_path = Path('/content/TFG')
	else:
		root_path = Path.home() / 'Documents' / 'TFG'
	
	data_folder = root_path / 'data'
	checkpoints_folder = root_path / 'checkpoints'
	checkpoints_folder.mkdir(exist_ok=True)

	# Get dataset info
	dataset_info_path = data_folder / 'dataset_info.pkl'
	with open(dataset_info_path, 'rb') as f:
		dataset_info = pickle.load(f)

	df_train = dataset_info['df_train']
	df_val = dataset_info['df_val']
	class_to_int = dataset_info['class_to_int']

	# Get datasets
	image_size = tuple(args.image_size)
	train_dataset = HAM10000Dataset(
		df_train,
		data_folder,
		image_size,
		args.padding
	)
	val_dataset = HAM10000Dataset(
		df_val,
		data_folder,
		image_size,
		args.padding
	)

	# Get dataloaders
	num_workers = os.cpu_count() - 1
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch,
		shuffle=True,
		num_workers=num_workers,
		# pin_memory=True
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch,
		shuffle=True,
		num_workers=num_workers,
		# pin_memory=True
	)
	
	# Get model
	in_channels = 3
	num_classes = len(class_to_int)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = UNet(
			in_channels=in_channels,
			num_classes=num_classes,
			padding=1 if args.padding else 0
		).to(device)

	base_model_path = checkpoints_folder / 'untrained_model.pth'
	if base_model_path.exists():
		model.load_state_dict(
			torch.load(
				base_model_path,
				weights_only=False,
				map_location=device
			)
		)
	else:
		# Save base untrained model
		torch.save(model.state_dict(), base_model_path)

	# Get Loss function and optimizer
	# criterion = nn.CrossEntropyLoss()
	criterion = DiceLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# Scheduler
	# TODO decidir si es vol fe servir un scheduler

	train_model(
		model= model,
		train_loader= train_loader,
		val_loader= val_loader,
		criterion= criterion,
		optimizer= optimizer,
		epochs=args.epochs,
		checkpoints_folder= checkpoints_folder,
		experiment=args.experiment
	)
