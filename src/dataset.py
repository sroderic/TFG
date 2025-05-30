import args
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm
from utils import get_images_folder, get_masks_folder


class HAM10000Dataset(Dataset):
	def __init__(self, df, data_folder):
		self.df = df
		self.images_folder = get_images_folder(data_folder)
		self.masks_folder = get_masks_folder(data_folder)
		self.images = []
		self.masks = []
		self.image_transform = transforms.Compose([
			# transforms.Resize(self.image_size),
			transforms.ToTensor(),
			transforms.ConvertImageDtype(dtype= torch.bfloat16),
			transforms.Normalize(mean=0.5, std=0.5),
			# transforms.Normalize(mean=[0.7565, 0.5503, 0.5762],
						# std=[0.1427, 0.1534, 0.1717])
			])
		self.mask_transform = transforms.Compose([
			# transforms.Resize(self.mask_size, interpolation=transforms.InterpolationMode.NEAREST),  # mantenim els valors discrets
			transforms.PILToTensor()
			])

		# To test

		for _, row in tqdm(df.iterrows(), total=len(df), desc="Preloading dataset to memory"):
			image_id = row['image_id']
			image, mask = self.__transform__(image_id)
			self.images.append(image.to(args.device))
			self.masks.append(mask.to(args.device))  # shape: (H, W) instead of (1, H, W)
		# self.images = torch.stack(images)
		# self.masks = torch.stack(masks) 
		# end test

	def __len__(self):
		return len(self.df)		
	
	def __getitem__(self, idx):
		# To test
		return self.images[idx], self.masks[idx]
		# end test
		image_id = self.df.iloc[idx]['image_id']
		return self.__transform__(image_id)
	
	def __transform__(self, image_id):
		image_file = self.images_folder / f"{image_id}.jpg"
		image = Image.open(image_file).convert('RGB')
		image = self.image_transform(image)

		mask_file = self.masks_folder / f"{image_id}.png"
		mask = Image.open(mask_file).convert('L')
		mask = self.mask_transform(mask)

		return image, mask.squeeze(0)