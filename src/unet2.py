import torch
import torch.nn as nn

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

		# convolution 3x3, ReLU (Downsampling)
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


	def __max_pool_1__(self, x):
		self.convolution_1 = self.double_convolution_down_1(x)
		return self.pool(self.convolution_1)
	
	def __max_pool_2__(self, x):
		self.convolution_2 = self.double_convolution_down_2(x)
		return self.pool(self.convolution_2)
	
	def __max_pool_3__(self, x):
		self.convolution_3 = self.double_convolution_down_3(x)
		return self.pool(self.convolution_3)
	
	def __max_pool_4__(self, x):
		self.convolution_4 = self.double_convolution_down_4(x)
		return self.pool(self.convolution_4)
	
	def __up_conv_4__(self, x):
		up_conv = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
		if self.padding == 0:
			crop = self.convolution_4.shape[2] // 2 - x.shape[2] 
			self.convolution_4 = self.convolution_4[:, :, crop:-crop, crop:-crop]
		return self.double_convolution_up_4(torch.cat([self.convolution_4, up_conv(x)], 1))

	def __up_conv_3__(self, x):
		up_conv = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		if self.padding == 0:
			crop = self.convolution_3.shape[2] // 2 - x.shape[2]
			self.convolution_3 = self.convolution_3[:, :, crop:-crop, crop:-crop]
		return self.double_convolution_up_3(torch.cat([self.convolution_3, up_conv(x)], 1))

	def __up_conv_2__(self, x):
		up_conv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		if self.padding == 0:
			crop = self.convolution_2.shape[2] // 2 - x.shape[2]
			self.convolution_2 = self.convolution_2[:, :, crop:-crop, crop:-crop]
		return self.double_convolution_up_2(torch.cat([self.convolution_2, up_conv(x)], 1))

	def __up_conv_1__(self, x):
		up_conv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		if self.padding == 0:
			crop = self.convolution_1.shape[2] // 2 - x.shape[2]
			self.convolution_1 = self.convolution_1[:, :, crop:-crop, crop:-crop]
		return self.double_convolution_up_1(torch.cat([self.convolution_1, up_conv(x)], 1))

	def forward(self, x):
		return self.out(
			self.__up_conv_1__(
				self.__up_conv_2__(
					self.__up_conv_3__(
						self.__up_conv_4__(
							self.bottle_neck(
								self.__max_pool_4__(
									self.__max_pool_3__(
										self.__max_pool_2__(
											self.__max_pool_1__(x)
										)
									)
								)
							)
						)
					)
				)
			)
		)

from torchinfo import summary
if __name__ == "__main__":
	# double_conv = DoubleConv(256, 256, 0)
	# print(double_conv)

	model = UNet(3, 10, 0)
	summary(model, input_size=(1, 3, 428, 572))

	# model = UNet(3, 10, 1)
	# summary(model, input_size=(1, 3, 608, 448))
	# summary(model, input_size=(1, 3, 416, 560))
