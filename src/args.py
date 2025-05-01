from argparse import ArgumentParser


def get_arguments():

	''' Defines and parses the command-line arguments
	
	'''

	parser = ArgumentParser()

	parser.add_argument('--colab', action='store_true')
	parser.add_argument('--name', type=str, required=True)
	parser.add_argument('--redux', type=int, required=True, help='UNet chamnnels reduction rate 2**n')
	parser.add_argument('--width', type=int, required=True, help='512 for 512x384, 384 for 384x288, 256 for 256x192')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--learning_rate' , '-lr', type=float, required=True)
	parser.add_argument('--loss', type=str, required=True, help='Cross, Dice, Focal0, Focal2, Focal3, FP')
	parser.add_argument('--optimizer', type=str, required=True, help='Adam, SGD')
	parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')	

	return parser.parse_args()
