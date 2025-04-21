import torch
import torch.nn as nn
from torchvision import transforms

class UNet(nn.Module):
	def __init__(self, in_channels, num_classes, padding):
		super().__init__()
		
		self.padding =padding

		# convolution 3x3, ReLU (Downsampling)
		self.double_convolution_down_1 = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		self.double_convolution_down_2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		self.double_convolution_down_3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		self.double_convolution_down_4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		# max pool 2x2
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# bottle_neck (conv 3x3, ReLU)
		self.bottle_neck = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		self.up_conv_4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  
		self.up_conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.up_conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.up_conv_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

		# convolution 3x3, ReLU (Upsampling)
		self.double_convolution_up_4 = nn.Sequential(
			nn.Conv2d(1024, 512, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		self.double_convolution_up_3 = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		self.double_convolution_up_2 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)
		
		self.double_convolution_up_1 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=self.padding),
			nn.ReLU(inplace=True)
		)

		self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)


	def forward(self, x):
		# Encoder
		x1 = self.double_convolution_down_1(x)
		x2 = self.double_convolution_down_2(self.pool(x1))
		x3 = self.double_convolution_down_3(self.pool(x2))
		x4 = self.double_convolution_down_4(self.pool(x3))

		
		# Bottleneck
		x = self.bottle_neck(self.pool(x4))
		
		# Decoder
		x = self.up_conv_4(x)
		if self.padding == 0:
			x4 = self.__center_crop__(x4, x)
		x = self.double_convolution_up_4(torch.cat([x4, x], dim=1))

		
		x = self.up_conv_3(x)
		if self.padding == 0:
			x3 = self.__center_crop__(x3, x)
		x = self.double_convolution_up_3(torch.cat([x3, x], dim=1))

		x = self.up_conv_2(x)
		if self.padding == 0:
			x2 = self.__center_crop__(x2, x)
		x = self.double_convolution_up_2(torch.cat([x2, x], dim=1))

		x = self.up_conv_1(x)
		if self.padding == 0:
			x1 = self.__center_crop__(x1, x)
		x = self.double_convolution_up_1(torch.cat([x1, x], dim=1))

		return self.out(x)

	def __center_crop__(self, tensor, target_tensor):
		_, _, h, w = target_tensor.shape
		transform = transforms.CenterCrop((h, w))
		return transform(tensor)



if __name__ == "__main__":
	from torchinfo import summary

	# double_conv = DoubleConv(256, 256, 0)
	# print(double_conv)

	model = UNet(3, 10, 0)
	summary(model, input_size=(1, 3, 428, 572))
	# model = UNet(3, 10, 1)
	# summary(model, input_size=(1, 3, 608, 448))
	# summary(model, input_size=(1, 3, 416, 560))
