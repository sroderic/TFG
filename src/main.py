import argparse
from pathlib import Path
import torch
import os
import pickle
from train import train
# import math
# import training_loop
# import test
# import inference

# # import threshold

def main(config):
	
	config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	config.num_workers = os.cpu_count()

	# Data folders
	if config.colab:
		config.root_path = Path('/content/TFG')
	else:
		config.root_path = Path.home() / 'Documents' / 'TFG'
	
	config.data_folder = config.root_path / 'data'

	# Image and mask size
	config.image_size = tuple(config.image_size)
	
	# config.experiment_path = config.data_path / 'models' / config.experiment / f'{config.sample_rate}_{config.window_size}_{config.mel_bands}'
  
	dataset_info_path = config.data_folder / 'dataset_info.pkl'
	with open(dataset_info_path, 'rb') as f:
		dataset_info = pickle.load(f)

	config.df_train = dataset_info['df_train']
	config.df_val = dataset_info['df_val']
	config.class_to_int = dataset_info['class_to_int']
	config.num_classes = len(config.class_to_int)
	
	if config.mode == 'train':
		# Train
		train(config)
#   elif config.mode == 'test':
#	 # Test
#	 test.test(
#	   config,
#	 )
#   elif config.mode == 'inference':
#	 # Test
#	 inference.infer(
#	   config,
#	 )
  



if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--preload', action='store_true')
	parser.add_argument('--colab', action='store_true')
	parser.add_argument('--mode', type=str, required=True, help='train, test, inference')
	parser.add_argument('--image_size', type=int, nargs=2, default=[428, 572], help='Input image size as two integers')
	parser.add_argument('--padding', action='store_true')
	parser.add_argument('--epochs', type=int, required=True)
	parser.add_argument('--batch', type=int, required=True)
	parser.add_argument('--lr', type=float, required=True)
	parser.add_argument('--optimizer', type=str, default='Adam', help='Adam, RMSprop, SGD')
	parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')	

	config = parser.parse_args()
	main(config)