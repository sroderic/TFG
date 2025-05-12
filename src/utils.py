import args
from argparse import ArgumentParser
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def set_arguments():

	''' Defines and parses the command-line arguments
	
	'''
	parser = ArgumentParser()

	parser.add_argument('--colab', action='store_true')
	parser.add_argument('--seed', type=int, required=True)
	parser.add_argument('--data_ratio', type=int, required=True)
	parser.add_argument('--features', type=int, required=True)
	# parser.add_argument('--name', type=str, required=True)
	# parser.add_argument('--redux', type=int, required=True, help='UNet channels reduction rate 2**n')
	parser.add_argument('--width', type=int, required=True, help='512 for 512x384, 384 for 384x288, 256 for 256x192')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--learning_rate' , '-lr', type=float, required=True)
	parser.add_argument('--loss', type=str, required=True, help='Combo, Cross, Dice, Focal0, Focal2, Focal3, Jaccard, LogCosh, Recall')
	parser.add_argument('--optimizer', type=str, required=True, help='Adam, SGD')
	parser.add_argument('--experiment', type=str, required=False, help='Name of the experiment')	
	parsed = parser.parse_args()

	args.colab = parsed.colab
	args.seed = parsed.seed
	args.data_ratio = parsed.data_ratio
	args.features = parsed.features
	args.width = parsed.width
	args.epochs = parsed.epochs
	args.batch_size = parsed.batch_size
	args.learning_rate= parsed.learning_rate
	args.loss= parsed.loss.lower()
	args.optimizer = parsed.optimizer.lower()
	args.experiment = parsed.experiment.lower()


def get_dataset_info(dataset_info_folder, seed):
	
	# Carreguem l'arxiu CSV
	csv_path = dataset_info_folder / 'HAM10000_metadata'
	df = pd.read_csv(csv_path)

	df = df.drop(columns=['dx_type', 'age', 'sex', 'localization', 'dataset'])
	classes = pd.unique(df['dx'])
	
	df_unique = df.groupby('lesion_id').sample(n=1, random_state=seed).reset_index(drop=True)
	df_train, df_val = train_test_split(
		df_unique,
		test_size=1 - (args.data_ratio * 0.01),
		stratify=df_unique['dx'],
		random_state=seed
	)	

	# Es crear un diccionari que asigna un valor numèric a cada classe de lesió
	class_to_int = {'background': 0 }
	for i, cls in enumerate(classes):
		class_to_int[cls] = i + 1

	
	int_to_class = {v: k for k, v in class_to_int.items()}
	return {
		"classes": classes,
		"class_to_int": class_to_int,
		"int_to_class": int_to_class,
		"df": df,
		"df_train": df_train,
		"df_val": df_val
	}

def get_dataset(df, data_folder):
	images_folder = get_images_folder(data_folder)
	masks_folder = get_masks_folder(data_folder)

	image_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.ConvertImageDtype(dtype= torch.bfloat16),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])
	
	mask_transform = transforms.Compose([
		transforms.PILToTensor()
		])

	images = []
	masks = []
	for _, row in tqdm(df.iterrows(), desc="Preloading dataset to memory"):
		image_id = row['image_id']
		
		image_file = images_folder / f"{image_id}.jpg"
		image = Image.open(image_file).convert('RGB')
		image = image_transform(image)
		image = image.to(args.device)
		images.append(image)

		mask_file = masks_folder / f"{image_id}.png"
		mask = Image.open(mask_file).convert('L')
		mask = mask_transform(mask)
		mask = mask.squeeze(0).to(args.device)
		masks.append(mask)
	
	images = torch.stack(images).to(device=args.device)
	masks = torch.stack(masks).to(device=args.device)
	
	del images_folder, masks_folder, image_transform, mask_transform, image_id, image_file, mask_file

	dataset = TensorDataset(images, masks)
	return dataset

def get_images_folder(data_folder):
	width = args.width
	height = int(width * 0.75)
	return data_folder / 'images' / f'{height}x{width}'

def get_masks_folder(data_folder):
	width = args.width
	height = int(width * 0.75)
	return data_folder / 'masks' / f'{height}x{width}'