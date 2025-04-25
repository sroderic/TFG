import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import dice_coeff

class DiceLoss(nn.Module):
	def __init__(self, smooth = 0.0):
		super(DiceLoss, self).__init__()
		self.smooth = smooth
		self.eps = 1e-7

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W]
		
		prob = F.softmax(logits, dim=1) # [N, C, H, W]
		
		num_classes = logits.size(1)  # Number of classes (C)
		
		target = F.one_hot(target, num_classes)  # [N, H, W] -> [N, H, W, C]
		target = target.permute(0, 3, 1, 2)  #  [N,HxW, C] -> [N, C, H, W]

		prob = prob.flatten(1)   # [N, C, H, W] -> [N, C x H x W]
		target = target.flatten(1).type_as(prob)   # [N, C, H, W] -> [N, C x H x W]
		
		dice =  dice_coeff(prob, target, self.smooth, self.eps)
		loss = 1.0 - dice

		return loss.mean()

class FocalLoss(nn.Module):
	def __init__(self):
		super(FocalLoss, self).__init__()
		self.gamma=2.0
		self.alpha=0.25

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W]
				
		# Compute standard cross-entropy
		ce = F.cross_entropy(logits, target, reduction='none')# [N, H, W]
		
		# Extract pt
		pt = ce.exp()

		# apply the Focal scaling factor
		loss = self.alpha * (1 - pt)**self.gamma * ce

		return loss.mean()