import time
import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from losses import DiceLoss, FocalLoss
from metrics import reset_confusion_matrix_elements, update_confusion_matrix_elements, calculate_metrics
def train_one_epoch(model, train_loader, criterion, optimizer):
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
	
	# To save the best model
	best_val_loss = float('inf')
	experiment_folder = checkpoints_folder / f'{experiment}'
	experiment_folder.mkdir(exist_ok=True)

	start_training = time.time()
	for epoch in range(epochs):
		metrics = {'epoch': epoch + 1}
		conf_matrices = { }
		print(f"📘 Epoch [{epoch+1}/{epochs}]")
		# Training
		model.train()
		avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
		metrics['train_loss'] = avg_train_loss
		print(f"   🟢 Train Loss: {avg_train_loss:.4f} -- Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}")
	
		# Validation	
		val_loss = 0.
		reset_confusion_matrix_elements(conf_matrices, num_classes)
		model = model.eval()
		with torch.no_grad():
			for images, masks in tqdm(val_loader, desc=f"Validation"):
				images = images.to(device)
				target = masks.long().to(device)  # (B, H, W)
		
				# forward
				logits = model(images)  # (B, C, H, W)
				loss = criterion(logits, target)
				val_loss += loss.item()
		
				prob = F.softmax(logits, dim=1) # [N, C, H, W] Probabilities
				pred = torch.argmax(prob, dim=1)  # [N, H, W] Class predictions
				update_confusion_matrix_elements(pred, target, conf_matrices, num_classes)
				
		
		avg_val_loss = val_loss / len(val_loader)
		metrics['val_loss'] = avg_train_loss
		calculate_metrics(metrics, conf_matrices, num_classes)

		print(f"   🔵 Val   Loss      : {avg_val_loss:.4f} -- Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}")
		print(f'   🔵 accuracy        : {metrics["accuracy"]:4f}')
		print(f'   🔵 precision       : {metrics["precision"]:4f}')
		print(f'   🔵 recall          : {metrics["recall"]:4f}')
		print(f'   🔵 f1              : {metrics["f1"]:4f}')
		print(f'   🔵 IoU             : {metrics["iou"]:4f}')
		print(f'   🔵 Dice            : {metrics["dice"]:4f}')

				
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
	from model import UNet, UNetRedux
	# from unet import UNet
	from dataset import HAM10000Dataset
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
	parser.add_argument('--model', type=str, required=True, help='UNet, UNetRedux')
	parser.add_argument('--image_size', type=int, required=True, help='512 for 512x384, 384 for 384x288, 256 for 256x192')
	# parser.add_argument('--padding', action='store_true')
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
		drive_path = Path('/content/drive/MyDrive/TFG')
		checkpoints_folder = drive_path / 'checkpoints'
		logs_folder = drive_path / 'logs'
	else:
		root_path = Path.home() / 'Documents' / 'TFG'
		checkpoints_folder = root_path / 'checkpoints'
		logs_folder = root_path / 'logs'
		
	data_folder = root_path / 'data'
	checkpoints_folder.mkdir(exist_ok=True)

	# Get dataset info
	dataset_info_path = data_folder / 'dataset_info.pkl'
	with open(dataset_info_path, 'rb') as f:
		dataset_info = pickle.load(f)

	df_train = dataset_info['df_train']
	df_val = dataset_info['df_val']
	class_to_int = dataset_info['class_to_int']

	# Get datasets
	image_size = (args.image_size*450//600, args.image_size)
	train_dataset = HAM10000Dataset(
		df_train,
		data_folder,
		image_size,
	)
	val_dataset = HAM10000Dataset(
		df_val,
		data_folder,
		image_size,
	)

	# Get dataloaders
	num_workers = os.cpu_count() - 1
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True
	)
	
	# Get model
	in_channels = 3
	num_classes = len(class_to_int)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if args.model == 'UNet':
		model = UNet(
			in_channels=in_channels,
			num_classes=num_classes,
			# padding=1 if args.padding else 0
		).to(device)
	elif args.model == 'UNetRedux':
		model = UNetRedux(
			in_channels=in_channels,
			num_classes=num_classes,
			# padding=1 if args.padding else 0
		).to(device)
	else:
		print('Optionss: UNet, UNetRedux')
		exit()
		

	# base_model_path = checkpoints_folder / 'untrained_model.pth'
	# if base_model_path.exists():
	# 	model.load_state_dict(
	# 		torch.load(
	# 			base_model_path,
	# 			weights_only=False,
	# 			map_location=device
	# 		)
	# 	)
	# else:
	# 	# Save base untrained model
	# 	torch.save(model.state_dict(), base_model_path)

	# Get Loss function and optimizer
	if args.loss == 'Cross':
		criterion = nn.CrossEntropyLoss()
	elif args.loss == 'Dice':
		criterion = DiceLoss()
	elif args.loss == 'Focal':
		criterion = FocalLoss()
	else:
		print('Options: Cross, Dice, Focal')
		exit()
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
