from args import get_arguments
from pathlib import Path
import pickle
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import random
import numpy as np

from dataset import HAM10000Dataset
from model import UNet
from losses import FocalLoss, FpFocalLoss
from metrics import Metrics
from train_supervised import train_model

# Get the arguments
args = get_arguments()

if __name__ == "__main__":

	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	# torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(mode=True, warn_only=True)
	torch.backends.cudnn.benchmark = False

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Data folders
	if args.colab:
		root_path = Path('/content/TFG')
		save_folder = Path('/content/drive/MyDrive/TFG')
	else:
		root_path = Path.home() / 'Documents' / 'TFG'
		save_folder = root_path
	
	data_folder = root_path / 'data'
	root_path.mkdir(exist_ok=True)
	save_folder.mkdir(exist_ok=True)
	
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
	
	base_model_path = save_folder / 'checkpoints' / f'untrained_{args.name.lower()}.pth'
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
	
	if args.loss.lower() == 'cross':
		criterion = nn.CrossEntropyLoss()
	elif args.loss.lower() == 'fp':
		criterion = FpFocalLoss()
	elif args.loss.lower() == 'focal0':
		criterion = FpFocalLoss(gamma=0.)
	elif args.loss.lower() == 'focal2':
		criterion = FocalLoss(gamma=2.)
	elif args.loss.lower() == 'focal3':
		criterion = FocalLoss(gamma=3.)
	else:
		print('Options: Cross, Focal, Fp')
		exit()
	
	if args.optimizer.lower() == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	elif args.optimizer.lower() == 'sgd':
		optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.99)
	else:
		print('Options: Adam, SGD')
		exit()
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
		name=args.name.lower(),
		experiment=args.experiment.lower(),
		device= device 
	)