import time
import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
# from sklearn import metrics
# import numpy as np

class DiceLoss(nn.Module):
	def __init__(self, smooth = 1.0):
		super(DiceLoss, self)
		self.smooth = smooth

	def forward(self, logits, targets):
		return self.__dice_loss__(logits, targets)
	
	def __dice_loss__(self, logits, targets):
		probs = F.softmax(logits, dim=1)  # [N, C, H, W]
		num_classes = probs.shape[1]  # Number of classes (C)
		
		targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2)  # [N,H,W] -> [N,H,W,C]
		
		intersection = (probs * targets).sum(dim=(2, 3))  # Element-wise multiplication [N,C]
		union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # Sum of all pixels [N,C]
			
		dice_score = (2. * intersection + self.smooth) / (union + self.smooth)  # [N, C]
		return 1 - dice_score.mean()  # Average Dice Loss across classes

def dice_coeff(logits, targets, smooth=1.0, return_per_class=False):
	"""
	Calculates Dice coefficient per class for multi-class segmentation.

	Args:
		pred:		[N, C, H, W] logits (not softmaxed).
		target:	  [N, H, W] ground truth class indices (ints from 0 to C-1).
		num_classes: int, number of classes.
		smooth:	  float, smoothing constant to avoid division by zero.
		return_per_class: if True, return per-class Dice scores. Else, return mean Dice.

	Returns:
		Dice score per class [C] or mean Dice (scalar)
	"""
	# Step 1: Convert logits to predicted probabilities
	probs = F.softmax(logits, dim=1)  # [N, C, H, W]
	num_classes = probs.shape[1]  # Number of classes (C)

	# Step 2: Get predicted class index per pixel
	pred_classes = torch.argmax(probs, dim=1)  # [N, H, W]

	# Step 3: One-hot encode both pred and target
	pred_onehot = F.one_hot(pred_classes, num_classes).permute(0, 3, 1, 2).float()   # [N, C, H, W]
	targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()	   # [N, C, H, W]

	# Step 4: Compute Dice per class
	intersection = (pred_onehot * targets_onehot).sum(dim=(2, 3))  # [N, C]
	union = pred_onehot.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(2, 3))  # [N, C]

	dice_per_image_per_class = (2 * intersection + smooth) / (union + smooth)  # [N, C]
	return dice_per_image_per_class
	dice_per_class = dice_per_image_per_class.mean(dim=0)

	if return_per_class:
		return dice_per_class  # [C]
	else:
		return dice_per_class.mean()  # scalar

def train_one_epoch(epoch):
	running_loss = 0.
	for images, masks in tqdm(train_loader, desc=f"Training (Epoch {epoch+1})"):
		images = images.to(device)
		targets = masks.long().to(device)  # (B, H, W)
		optimizer.zero_grad()
		logits = model(images)  # (B, C, H, W)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	return running_loss / len(train_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):

	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Metrics per epoch
	logger_train_loss = []
	logger_validation_loss = []

	logger_dice_coeff = []
	
	# To save the best model
	best_val_loss = float('inf')
	# TODO afegir ruta i crear carpetes

	start_training = time.time()
	for epoch in range(epochs):

		# Training
		# model.train()
		# train_loss = train_one_epoch(epoch)
		# logger_train_loss.append(train_loss)
		train_loss = 10000
		print(f"\tLoss: {train_loss:4f} -- Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}")
		print(f"Validation (Epoch {epoch+1})")
	
		
		# Validation
		# TODO validation()
				
		start_validation = time.time()
		validation_loss = 0.
		validation_dice = []
		model = model.eval()
		with torch.no_grad():
			for images, masks in (val_loader):
				images = images.to(device)
				targets = masks.long().to(device)  # (B, H, W)
		
				# forward
				logits = model(images)  # (B, C, H, W)
				loss = criterion(logits, targets)
		
				validation_loss += loss.item()
				dice_batch = dice_coeff(logits= logits, targets=targets) # [B, C]
		
				validation_dice.append(dice_batch)
		validation_dice = torch.cat(validation_dice, dim =0)
		average_validation_loss = validation_loss / len(val_loader)
		average_dice =validation_dice.mean(dim=0).item()
		
		logger_validation_loss.append(average_validation_loss)
		logger_dice_coeff.append(average_dice)
		print(f'\tLoss: {average_validation_loss:4f} -- Time: {datetime.timedelta(seconds=time.time()-start_validation)}, Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}')
		print(f'\tDice Coefficient: {average_dice:4f}')

				
		# TODO save best model()

		# if (1 - average_validation_loss) > best_validation_loss:
		# 	best_validation_loss = 1 - average_validation_loss
		# 	best_model = {
		# 		'epoch': epoch + 1,
		# 		'model_state_dictionary': model.state_dict(),
		# 		'fixed_threshold': fixed_threshold,
		# 		'best_label_threshold': best_label_threshold,
		# 	}
		# 	best_model_probabilities = {
		# 		'probabilities': probabilities,
		# 		'ground_truths': ground_truths
		# 	}
		# 	best_model_file = config.experiment_path / 'best_model.pth'
		# 	torch.save(best_model, best_model_file)
		# 	print('Best validation loss!')
		# 	best_model_probabilities_file = config.experiment_path / 'best_model_probabilities.pth'
		# 	torch.save(best_model_probabilities, best_model_probabilities_file)
		print('........................................................................')

		# epoch_checkpoint = {
		# 		'epoch': epoch + 1,
		# 		'batch_size': config.batch,
		# 		'model_state_dictionary': model.state_dict(),
		# 		'optimizer_state_dictionary': optimizer.state_dict(),
		# 		'scheduler_state_dictionary': scheduler.state_dict(),
		# 		'logger_train_loss': logger_train_loss,
		# 		'logger_validation_loss': logger_validation_loss,
		# 		'roc_auc_macros': roc_auc_macros,
		# 		'pr_auc_macros': pr_auc_macros,
		# 		'f1_macros': f1_macros,
		# }
		# # model_checkpoint_file = config.experiment_path / 'checkpoint'
		# torch.save(epoch_checkpoint, model_checkpoint_file)
		# epoch_checkpoint_file = config.experiment_path / f'epoch{epoch + 1:02d}'
		# torch.save(epoch_checkpoint, epoch_checkpoint_file)

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
		epochs=args.epochs
	)
