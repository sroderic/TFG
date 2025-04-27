import torch
import torch.nn as nn
from torchvision import transforms

class UNet(nn.Module):
	def __init__(self, in_channels, num_classes, redux):
		super().__init__()
		features = 64 // redux

		# Encoder 1 - Double Convolution 3x3, ReLU (Downsampling) + max pool 2x2
		self.double_convolution_down_1 = nn.Sequential(
			nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
			nn.BatchNorm2d(features),
			nn.ReLU(),
			nn.Conv2d(features, features, kernel_size=3, padding=1),
			nn.BatchNorm2d(features),
			nn.ReLU()
		)
		self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Encoder 2
		self.double_convolution_down_2 = nn.Sequential(
			nn.Conv2d(features, features *2, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *2),
			nn.ReLU(),
			nn.Conv2d(features *2, features *2, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *2),
			nn.ReLU()
		)
		self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Encoder 3
		self.double_convolution_down_3 = nn.Sequential(
			nn.Conv2d(features *2, features *4, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *4),
			nn.ReLU(),
			nn.Conv2d(features *4, features *4, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *4),
			nn.ReLU()
		)
		self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Encoder 4
		self.double_convolution_down_4 = nn.Sequential(
			nn.Conv2d(features *4, features *8, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *8),
			nn.ReLU(),
			nn.Conv2d(features *8, features *8, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *8),
			nn.ReLU()
		)
		self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		# bottle_neck (conv 3x3, ReLU)
		self.bottle_neck = nn.Sequential(
			nn.Conv2d(features *8, features *16, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *16),
			nn.ReLU(),
			nn.Conv2d(features *16, features *16, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *16),
			nn.ReLU()
		)

		# Decoder 4 - Up convolution + Double convolution 3x3, ReLU (Upsampling)
		self.up_conv_4 = nn.ConvTranspose2d(features *16, features *8, kernel_size=2, stride=2)
		self.double_convolution_up_4 = nn.Sequential(
			nn.Conv2d(features *16, features *8, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *8),
			nn.ReLU(),
			nn.Conv2d(features *8, features *8, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *8),
			nn.ReLU()
		)

		# Decoder 3		
		self.up_conv_3 = nn.ConvTranspose2d(features *8, features *4, kernel_size=2, stride=2)
		self.double_convolution_up_3 = nn.Sequential(
			nn.Conv2d(features *8, features *4, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *4),
			nn.ReLU(),
			nn.Conv2d(features *4, features *4, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *4),
			nn.ReLU()
		)


		# Decoder 2
		self.up_conv_2 = nn.ConvTranspose2d(features *4, features *2, kernel_size=2, stride=2)
		self.double_convolution_up_2 = nn.Sequential(
			nn.Conv2d(features *4, features *2, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *2),
			nn.ReLU(),
			nn.Conv2d(features *2, features *2, kernel_size=3, padding=1),
			nn.BatchNorm2d(features *2),
			nn.ReLU()
		)

		# Decoder 1
		self.up_conv_1 = nn.ConvTranspose2d(features *2, features, kernel_size=2, stride=2)
		self.double_convolution_up_1 = nn.Sequential(
			nn.Conv2d(features *2, features, kernel_size=3, padding=1),
			nn.BatchNorm2d(features),
			nn.ReLU(),
			nn.Conv2d(features, features, kernel_size=3, padding=1),
			nn.BatchNorm2d(features),
			nn.ReLU()
		)

		self.out = nn.Conv2d(in_channels=features, out_channels=num_classes, kernel_size=1)

	def forward(self, x):
		# Encoder
		x1 = self.double_convolution_down_1(x)
		x2 = self.double_convolution_down_2(self.pool_1(x1))
		x3 = self.double_convolution_down_3(self.pool_2(x2))
		x4 = self.double_convolution_down_4(self.pool_3(x3))

		# Bottleneck
		x = self.bottle_neck(self.pool_4(x4))
		
		# Decoder
		x = self.up_conv_4(x)
		x = self.double_convolution_up_4(torch.cat([x4, x], dim=1))
		
		x = self.up_conv_3(x)
		x = self.double_convolution_up_3(torch.cat([x3, x], dim=1))

		x = self.up_conv_2(x)
		x = self.double_convolution_up_2(torch.cat([x2, x], dim=1))

		x = self.up_conv_1(x)
		x = self.double_convolution_up_1(torch.cat([x1, x], dim=1))

		self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)


		return self.out(x)
	

# class UNetRedux(nn.Module):
# 	def __init__(self, in_features, num_classes):
# 	# def __init__(self, in_features, num_classes, padding):
# 		super().__init__()
		
# 		self.padding =1

# 		# convolution 3x3, ReLU (Downsampling)
# 		self.double_convolution_down_1 = nn.Sequential(
# 			nn.Conv2d(in_features, 64, kernel_size=3, padding=self.padding),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, kernel_size=3, padding=self.padding),
# 			nn.ReLU()
# 		)

# 		self.double_convolution_down_2 = nn.Sequential(
# 			nn.Conv2d(64, 128, kernel_size=3, padding=self.padding),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, kernel_size=3, padding=self.padding),
# 			nn.ReLU()
# 		)

# 		self.double_convolution_down_3 = nn.Sequential(
# 			nn.Conv2d(128, 256, kernel_size=3, padding=self.padding),
# 			nn.ReLU(),
# 			nn.Conv2d(256, 256, kernel_size=3, padding=self.padding),
# 			nn.ReLU()
# 		)

# 		# max pool 2x2
# 		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 		# bottle_neck (conv 3x3, ReLU)
# 		self.bottle_neck = nn.Sequential(
# 			nn.Conv2d(256, 512, kernel_size=3, padding=self.padding),
# 			nn.ReLU(),
# 			nn.Conv2d(512, 512, kernel_size=3, padding=self.padding),
# 			nn.ReLU()
# 		)

# 		self.up_conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
# 		self.up_conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
# 		self.up_conv_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

# 		# convolution 3x3, ReLU (Upsampling)

# 		self.double_convolution_up_3 = nn.Sequential(
# 			nn.Conv2d(512, 256, kernel_size=3, padding=self.padding),
# 			nn.ReLU(),
# 			nn.Conv2d(256, 256, kernel_size=3, padding=self.padding),
# 			nn.ReLU()
# 		)

# 		self.double_convolution_up_2 = nn.Sequential(
# 			nn.Conv2d(256, 128, kernel_size=3, padding=self.padding),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, kernel_size=3, padding=self.padding),
# 			nn.ReLU()
# 		)
		
# 		self.double_convolution_up_1 = nn.Sequential(
# 			nn.Conv2d(128, 64, kernel_size=3, padding=self.padding),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, kernel_size=3, padding=self.padding),
# 			nn.ReLU()
# 		)

# 		self.out = nn.Conv2d(in_features=64, out_features=num_classes, kernel_size=1)


# 	def forward(self, x):
# 		# Encoder
# 		x1 = self.double_convolution_down_1(x)
# 		x2 = self.double_convolution_down_2(self.pool(x1))
# 		x3 = self.double_convolution_down_3(self.pool(x2))
		
# 		# Bottleneck
# 		x = self.bottle_neck(self.pool(x3))
		
# 		# Decoder
# 		x = self.up_conv_3(x)
# 		if self.padding == 0:
# 			x3 = self.__center_crop__(x3, x)
# 		x = self.double_convolution_up_3(torch.cat([x3, x], dim=1))

# 		x = self.up_conv_2(x)
# 		if self.padding == 0:
# 			x2 = self.__center_crop__(x2, x)
# 		x = self.double_convolution_up_2(torch.cat([x2, x], dim=1))

# 		x = self.up_conv_1(x)
# 		if self.padding == 0:
# 			x1 = self.__center_crop__(x1, x)
# 		x = self.double_convolution_up_1(torch.cat([x1, x], dim=1))

# 		return self.out(x)

# 	def __center_crop__(self, tensor, target_tensor):
# 		_, _, h, w = target_tensor.shape
# 		transform = transforms.CenterCrop((h, w))
# 		return transform(tensor)


# if __name__ == "__main__":
# 	from torchinfo import summary

# 	model = UNet(3, 8, redux=2)
# 	summary(model, input_size=(1, 3, 192, 256))

# 	model = UNet(3, 8, redux=1)
# 	summary(model, input_size=(1, 3, 192, 256))
# 	# summary(model, input_size=(1, 3, 416, 560))


# 	model = UNet(3, 8, redux=0)
# 	summary(model, input_size=(1, 3, 192, 256))
# 	# summary(model, input_size=(1, 3, 416, 560))
