import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

class HAM10000Dataset(Dataset):
	def __init__(self, df, images_folder, masks_folder, image_transform=None, mask_transform=None):
		self.df = df
		self.images_folder = images_folder
		self.masks_folder = masks_folder
		self.image_transform = image_transform
		self.mask_transform = mask_transform

	def __len__(self):
		return len(self.df)		
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		image_id = row['image_id']

		image_path = self.images_folder / f"{image_id}.jpg"
		mask_path = self.masks_folder / f"{image_id}.png"

		image = Image.open(image_path)
		mask = Image.open(mask_path)

		if self.image_transform:
			image = self.image_transform(image)
		if self.mask_transform:
			mask = self.mask_transform(mask)

		return image, mask




import utils
import matplotlib.pyplot as plt
if __name__ == "__main__":

	dataset = HAM10000Dataset(
		utils.df_train,
		utils.images_folder,
		utils.masks_folder,
		utils.image_transform,
		utils.mask_transform
		)

	# Elegir un índice cualquiera (por ejemplo, el primero)
	image, mask = dataset[5000]

	# Convertir el tensor de imagen a numpy para mostrarlo (des-normalizando)
	mean = torch.tensor([0.7565, 0.5503, 0.5762]).view(3, 1, 1)
	std = torch.tensor([0.1427, 0.1534, 0.1717]).view(3, 1, 1)
	image_np = image * std + mean  # Desnormalizar
	image_np = image_np.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)

	# Asegurarse que la máscara es 2D
	mask_np = mask.squeeze().numpy()

	# Mostrar imagen y máscara
	plt.figure(figsize=(12, 6))

	plt.subplot(1, 2, 1)
	plt.imshow(image_np)
	plt.title("Imagen")

	plt.subplot(1, 2, 2)
	plt.imshow(mask_np, cmap='tab10')
	plt.title("Máscara (Segmentación Semántica)")

	plt.tight_layout()
	plt.show()


	# def show_batch(dataset, n=9):
	# 	rows = cols = int(n**0.5)
	# 	fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 4))
		
	# 	mean = torch.tensor([0.7565, 0.5503, 0.5762]).view(3, 1, 1)
	# 	std = torch.tensor([0.1427, 0.1534, 0.1717]).view(3, 1, 1)

	# 	for i in range(n):
	# 		image, mask = dataset[i]
	# 		image = image * std + mean  # Desnormalizar
	# 		image = image.permute(1, 2, 0).numpy()
	# 		mask = mask.squeeze().numpy()

	# 		row = i // cols
	# 		col = (i % cols) * 2

	# 		axes[row, col].imshow(image)
	# 		axes[row, col].set_title("Imagen")
	# 		axes[row, col].axis('off')

	# 		axes[row, col + 1].imshow(mask, cmap='tab10')
	# 		axes[row, col + 1].set_title("Máscara")
	# 		axes[row, col + 1].axis('off')

	# plt.tight_layout()
	# plt.show()

	# show_batch(dataset, n=9)