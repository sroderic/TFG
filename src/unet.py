import torch.nn as nn
from unet_parts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
	def __init__(self, in_channels, num_classes):
		padding = 1
		super().__init__()
		self.down_convolution_1 = DownSample(in_channels, 64, padding)
		self.down_convolution_2 = DownSample(64, 128, padding)
		self.down_convolution_3 = DownSample(128, 256, padding)
		self.down_convolution_4 = DownSample(256, 512, padding)

		self.boottle_neck = DoubleConv(512, 1024, padding)

		self.up_convolution_1 = UpSample(1024, 512, padding)
		self.up_convolution_2 = UpSample(512, 256, padding)
		self.up_convolution_3 = UpSample(256, 128, padding)
		self.up_convolution_4 = UpSample(128, 64, padding)

		self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

	def forward(self, x):
		down_1, p1 = self.down_convolution_1(x)
		down_2, p2 = self.down_convolution_2(p1)
		down_3, p3 = self.down_convolution_3(p2)
		down_4, p4 = self.down_convolution_4(p3)

		b = self. boottle_neck(p4)

		up_1 = self.up_convolution_1(b, down_4)
		up_2 = self.up_convolution_2(up_1, down_3)
		up_3 = self.up_convolution_3(up_2, down_2)
		up_4 = self.up_convolution_4(up_3, down_1)

		out = self.out(up_4)

		return out
	

if __name__ == "__main__":
	from torchinfo import summary

	# double_conv = DoubleConv(256, 256, 0)
	# print(double_conv)

	# model = UNet(3, 10, 0)
	# summary(model, input_size=(1, 3, 428, 572))
	model = UNet(3, 10)
	summary(model, input_size=(1, 3, 192, 256))
	# summary(model, input_size=(1, 3, 416, 560))