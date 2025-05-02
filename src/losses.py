import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
	def __init__(self, smooth = 1.0):
		super(DiceLoss, self).__init__()
		self.smooth = smooth

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W]

		
		prob = F.softmax(logits, dim=1) # [N, C, H, W]
		num_classes = logits.size(1)  # Number of classes (C)
		prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
		target = target.view(-1)
		
		dice = []
		for c in range(num_classes):
			target_c = (target == c).float()
			prob_c = prob[:, c]
			
			intersection = (prob_c * target_c).sum()
			dice_c = (2.*intersection + self.smooth)/(prob_c.sum() + target_c.sum() + self.smooth)
			dice.append(dice_c)
		dice =  torch.stack(dice)
		dice_loss = 1 - dice.mean()
		return dice_loss
	
class JaccardLoss(nn.Module):
	def __init__(self, smooth = 1.0):
		super(JaccardLoss, self).__init__()
		self.smooth = smooth

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W]

		
		prob = F.softmax(logits, dim=1) # [N, C, H, W]
		num_classes = logits.size(1)  # Number of classes (C)
		prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
		target = target.view(-1)
		
		dice = []
		for c in range(num_classes):
			target_c = (target == c).float()
			prob_c = prob[:, c]
			
			intersection = (prob_c * target_c).sum()
			dice_c = (intersection + self.smooth)/(prob_c.sum() + target_c.sum() - intersection + self.smooth)
			dice.append(dice_c)
		dice =  torch.stack(dice)
		dice_loss = 1 - dice.mean()
		return dice_loss

class LogCoshDiceLoss(nn.Module):
	def __init__(self, smooth = 1.0):
		super(LogCoshDiceLoss, self).__init__()
		self.smooth = smooth

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W].
		prob = F.softmax(logits, dim=1)  # [N, C, H, W]
		
		num_classes = logits.size(1)  # Number of classes (C)
		prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
		target = target.view(-1)
		
		dice = []
		for c in range(num_classes):
			target_c = (target == c).float()
			prob_c = prob[:, c]
			
			intersection = (prob_c * target_c).sum()
			dice_c = (2.*intersection + self.smooth)/(prob_c.sum() + target_c.sum() + self.smooth)
			dice.append(dice_c)
		dice =  torch.stack(dice)
		dice_loss = 1 - dice.mean()
		return torch.log(torch.cosh(dice_loss))

class FpFocalLoss(nn.Module):
	def __init__(self, gamma=2.0, alpha = 1.0):
		super(FpFocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W]
		# alpha [C]

		if isinstance(self.alpha, torch.Tensor):
			alpha_term = self.alpha[target]
		else:
			alpha_term = self.alpha

		# Compute standard cross-entropy
		ce = F.cross_entropy(logits, target, reduction='none')# [N, H, W]
		
		# Extract pt (CE = -log(pt) -> pt = exp(-CE))
		pt = (-ce).exp()

		# apply the Focal scaling factor
		focal_term = alpha_term * (1. - pt)**self.gamma
		focal_loss = focal_term * ce

		# Compute false-positive penalty for background (class 0)
		p = F.softmax(logits, dim=1)  # [N, C, H, W]
		p_bg = p[:, 0, :, :]  # probability of background class

		# Penalize when ground truth is NOT background
		non_bg_mask = (target != 0).type_as(p_bg)  # [N, H, W]
		fp_penalty = -(1. - p_bg).log() * non_bg_mask  # squared prob for smoothness

		loss = focal_loss + fp_penalty

		return loss.mean()
	
	def set_alpha(self, alpha):
		print('New alpha:', alpha)
		self.alpha = alpha

class FocalLoss(nn.Module):
	def __init__(self, gamma=2.0, alpha = 1.0):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W]
		# alpha [C]

		if isinstance(self.alpha, torch.Tensor):
			alpha_term = self.alpha[target]
		else:
			alpha_term = self.alpha

		# Compute standard cross-entropy
		ce = F.cross_entropy(logits, target, reduction='none')# [N, H, W]
		
		# Extract pt (CE = -log(pt) -> pt = exp(-CE))
		pt = (-ce).exp()

		# apply the Focal scaling factor
		focal_term = alpha_term * (1 - pt)**self.gamma
		loss = focal_term * ce

		return loss.mean()
	
	def set_alpha(self, alpha):
		print('New alpha:', alpha)
		self.alpha = alpha

class WeightedLoss(nn.Module):
	def __init__(self):
		super(WeightedLoss, self).__init__()

	def forward(self, logits, target):

		# logits [N, C, H, W]
		# target [N, H, W]
		# alpha [C]

		# Compute standard cross-entropy
		ce = F.cross_entropy(logits, target, reduction='none')# [N, H, W]
		
		# Extract pt (CE = -log(pt) -> pt = exp(-CE))
		pt = (-ce).exp()

		# apply the Focal scaling factor
		weight_term = -(pt + 1e-7).log() / torch.tensor(100).log()
		loss = weight_term * ce

		return loss.mean()