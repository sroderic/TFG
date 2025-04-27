from args import get_arguments
from pathlib import Path
import pickle
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from dataset import HAM10000Dataset
from model import UNet
from losses import DiceLoss, FocalLoss
from metrics import Metrics
from train_supervised import train_model

# Get the arguments
args = get_arguments()

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Data folders
	if args.colab:
		root_path = Path('/content/TFG')
		save_folder = Path('/content/drive/MyDrive/TFG')
	else:
		root_path = Path.home() / 'Documents' / 'TFG'
		save_folder = root_path
	
	data_folder = root_path / 'data'

	# Get dataset info
	dataset_info_path = data_folder / 'dataset_info.pkl'
	with open(dataset_info_path, 'rb') as f:
		dataset_info = pickle.load(f)

	df_train = dataset_info['df_train']
	df_val = dataset_info['df_val']
	class_to_int = dataset_info['class_to_int']

	# Get datasets
	image_size = (args.width*450//600, args.width)
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
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=num_workers,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=num_workers,
	)

	# Get model
	in_channels = 3
	num_classes = len(class_to_int)

	model = UNet(
			in_channels=in_channels,
			num_classes=num_classes,
			redux=2**args.redux
		).to(device)
	
	if args.loss == 'Cross':
		criterion = nn.CrossEntropyLoss()
	elif args.loss == 'Dice':
		criterion = DiceLoss()
	elif args.loss == 'Focal':
		criterion = FocalLoss()
	else:
		print('Options: Cross, Dice, Focal')
		exit()
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	metrics = Metrics(num_classes)

	train_model(
		model= model,
		train_loader= train_loader,
		val_loader= val_loader,
		criterion= criterion,
		optimizer= optimizer,
		epochs=args.epochs,
		metrics=metrics,
		save_folder= save_folder,
		experiment=args.name.lower(),
		device= device 
	)