import torch
import torch.nn as nn


class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		padding = 1
		self.conv_op = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.conv_op(x)

class DownSample(nn.Module):
	def __init__(self, in_channels, out_channels, padding):
		super().__init__()
		self.conv = DoubleConv(in_channels, out_channels, padding)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


	def forward(self,x):
		down = self.conv(x)
		p = self.pool(down)
		
		return down, p
	
class UpSample(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.padding = 1
		self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels, self.padding)

	def forward(self, x1, x2):
		
		x1 = self.up(x1)
		if self.padding == 0:
			crop = (x2.shape[2] - x1.shape[2]) // 2
			x2 = x2[:, :, crop:-crop, crop:-crop]
		x = torch.cat([x2, x1], 1)
		return self.conv(x)