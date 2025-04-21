from torchvision import transforms
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm

class HAM10000Dataset(Dataset):
	def __init__(self, df, data_folder, image_size, padding):
		self.df = df
		self.images_folder = data_folder / 'images'
		self.masks_folder = data_folder / 'semantic_segmentations'
		self.image_size = image_size
		self.mask_size = image_size if padding else tuple(x - 184 for x in image_size)

		self.image_transform = transforms.Compose([
			transforms.Resize(self.image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.7565, 0.5503, 0.5762],
						std=[0.1427, 0.1534, 0.1717])
			])
		self.mask_transform = transforms.Compose([
			transforms.Resize(self.mask_size, interpolation=transforms.InterpolationMode.NEAREST),  # mantenim els valors discrets
			transforms.PILToTensor()
			])
		
	def __len__(self):
		return len(self.df)		
	
	def __getitem__(self, idx):
			image_id = self.df.iloc[idx]['image_id']
			return self.__transform__(image_id)
	
	def __transform__(self, image_id):
		image_path = self.images_folder / f"{image_id}.jpg"
		mask_path = self.masks_folder / f"{image_id}.png"
		image = Image.open(image_path)
		mask = Image.open(mask_path)
		image = self.image_transform(image)
		
		mask = self.mask_transform(mask)
		return image, mask.squeeze(0)