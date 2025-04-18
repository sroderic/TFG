from pathlib import Path
import pickle
import pandas as pd
import torch
from torchvision import transforms

# IMAGE_SIZE = (450, 600)
# IMAGE_SIZE = (448, 608)
# MASK_SIZE = IMAGE_SIZE
IMAGE_SIZE = (428, 572)
MASK_SIZE = (244, 388)
PROJECT_FOLDER = Path.home() / 'Documents' / 'TFG'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder = PROJECT_FOLDER / 'data'
images_folder = data_folder / 'images'
masks_folder = data_folder / 'semantic_segmentations'

dataset_info_path = data_folder / 'dataset_info.pkl'
with open(dataset_info_path, 'rb') as f:
	dataset_info = pickle.load(f)

df_train = dataset_info['df_train']
df_val = dataset_info['df_val']
class_to_int = dataset_info['class_to_int']

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7565, 0.5503, 0.5762],
                         std=[0.1427, 0.1534, 0.1717])
])

mask_transform = transforms.Compose([
    transforms.Resize(MASK_SIZE, interpolation=transforms.InterpolationMode.NEAREST),  # mantenim els valors discrets
    transforms.PILToTensor()
])

# criterion
# optimizer