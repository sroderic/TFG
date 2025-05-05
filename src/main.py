import args
from pathlib import Path
import pickle
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import random
import numpy as np

from utils import set_arguments, get_dataset_info
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
	dataset_info_file = dataset_info_folder/ f'dataset_info_{args.seed}.pkl' # Nom depenent de la llavor, ha de contenir int_to_class i df_test/df_val. Si no existeix, s'ha de generar amb la llavor
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
			features=args.features
		).to(args.device)
	
	model_base_folder = args.save_folder / 'checkpoints' / f'UNet{args.features}'
	model_base_folder.mkdir(exist_ok=True)
	model_folder = model_base_folder / f'{args.seed}'
	model_folder.mkdir(exist_ok=True)
	model_file = model_folder / 'untrained.pth'

	if model_file.exists():
		model.load_state_dict(
			torch.load(
				model_file,
				weights_only=False,
				map_location=args.device
			)
		)
	else:
		# Save base untrained model
		torch.save(model.state_dict(), model_file)
	

	if args.loss.lower() == 'combo':
		criterion = ComboDiceLoss()
	elif args.loss.lower() == 'cross':
		criterion = nn.CrossEntropyLoss()
	elif args.loss.lower() == 'dice':
		criterion = DiceLoss()
	elif args.loss.lower() == 'focal0':
		criterion = FocalLoss(gamma=0.)
	elif args.loss.lower() == 'focal2':
		criterion = FocalLoss(gamma=2.)
	elif args.loss.lower() == 'focal3':
		criterion = FocalLoss(gamma=3.)
	elif args.loss.lower() == 'jaccard':
		criterion = JaccardLoss()
	elif args.loss.lower() == 'recall':
		criterion = RecallCrossEntropy()
	else:
		print('Options: Cross, Dice, Focal0, Focal2, Focal3, Jaccard, LogCosh, Recall')
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
		metrics=metrics
	)