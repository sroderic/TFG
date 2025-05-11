import args
from pathlib import Path
import pickle
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import random
import numpy as np

from utils import set_arguments, get_dataset_info, get_dataset
from dataset import HAM10000Dataset
from model import UNet
from losses import ComboDiceLoss, DiceLoss, FocalLoss, JaccardLoss, RecallCrossEntropy
from metrics import Metrics
from train_supervised import train_model

# Get the arguments
set_arguments()

if __name__ == "__main__":


	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	# torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(mode=True, warn_only=True)
	torch.backends.cudnn.benchmark = False

	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Data folders
	if args.colab:
		root_path = Path('/content/TFG')
		args.save_folder = Path('/content/drive/MyDrive/TFG')
	else:
		root_path = Path.home() / 'Documents' / 'TFG'
		args.save_folder = root_path
	
	root_path.mkdir(exist_ok=True)
	args.save_folder.mkdir(exist_ok=True)
	args.data_folder = root_path / 'data'

	# Get dataset info
	dataset_info_folder = args.save_folder / 'data' / 'ham10000'
	dataset_info_file = dataset_info_folder/ f'dataset_info_{args.seed}_{args.data_ratio}.pkl'
	if dataset_info_file.exists():
		with open(dataset_info_file, 'rb') as f:
			dataset_info = pickle.load(f)
	else:
		dataset_info = get_dataset_info(dataset_info_folder, args.seed)
		with open(dataset_info_file, "wb") as f:
			pickle.dump(dataset_info, f)
	
	df_train = dataset_info['df_train']
	df_val = dataset_info['df_val']
	class_to_int = dataset_info['class_to_int']

	dataset_folder = args.data_folder / 'sl'

	# Get tensordatasets
	train_dataset = get_dataset(df_train,
      dataset_folder)

	val_dataset = get_dataset(df_val,
		dataset_folder)
	
	# Get dataloaders
	num_workers = 0
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=num_workers,
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=num_workers,
	)
	
	'''
	# Get datasets
	train_dataset = HAM10000Dataset(
		df_train,
		dataset_folder,
	)
	val_dataset = HAM10000Dataset(
		df_val,
		dataset_folder,
	)
	# Get dataloaders
	num_workers = 4
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=num_workers,
	)
	'''
	del f, dataset_info, df_train, df_val
	
	
	# Get model
	in_channels = 3
	num_classes = len(class_to_int)

	model = UNet(
			in_channels=in_channels,
			num_classes=num_classes,
			features=args.features
		).to(device=args.device)
	

	if args.loss == 'combo':
		criterion = ComboDiceLoss()
	elif args.loss == 'cross':
		criterion = nn.CrossEntropyLoss()
	elif args.loss == 'dice':
		criterion = DiceLoss()
	elif args.loss == 'focal0':
		criterion = FocalLoss(gamma=0.)
	elif args.loss == 'focal2':
		criterion = FocalLoss(gamma=2.)
	elif args.loss == 'focal3':
		criterion = FocalLoss(gamma=3.)
	elif args.loss == 'jaccard':
		criterion = JaccardLoss()
	elif args.loss == 'recall':
		criterion = RecallCrossEntropy()
	else:
		print('Options: Cross, Dice, Focal0, Focal2, Focal3, Jaccard, LogCosh, Recall')
		exit()
	
	if args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.5)
	elif args.optimizer == 'sgd':
		optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.99)
	else:
		print('Options: Adam, SGD')
		exit()
	
	# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
	scheduler = None
	metrics = Metrics(num_classes)

	train_model(
		model= model,
		train_loader= train_loader,
		val_loader= val_loader,
		criterion= criterion,
		optimizer= optimizer,
		scheduler= scheduler,
		epochs=args.epochs,
		metrics=metrics
	)

# del train_loader, train_dataset, model, criterion