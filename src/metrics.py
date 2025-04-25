import torch

def dice_coeff(pred, target, smooth=0.0, eps = 1e-7):
	# logits [N, C, H, W]
	# target [N, C, H, W]
	
	intersection = torch.sum(pred * target, dim=1)
	sum_sets = torch.sum(pred + target, dim=1)
	dice_score = (2 * intersection + smooth) / (sum_sets + smooth).clamp_min(eps)
	
	return dice_score # [N]

def reset_confusion_matrix_elements(conf_matrices, num_classes):
	for cls in range(0, num_classes):
		conf_matrix = { }
		conf_matrix['TP'] = 0
		conf_matrix['TN'] = 0
		conf_matrix['FP'] = 0
		conf_matrix['FN'] = 0
		conf_matrices[f'class_{cls}'] = conf_matrix

def update_confusion_matrix_elements(pred, target, conf_matrices, num_classes):
	
	for cls in range(0, num_classes):
		pred_cls = (pred == cls).bool()
		target_cls = (target == cls).bool()

		# Compute TP, FP, FN, TN
		TP = (pred_cls & target_cls).sum().item()
		TN = ((~pred_cls) & (~target_cls)).sum().item()
		FP = (pred_cls & (~target_cls)).sum().item()
		FN = ((~pred_cls) & target_cls).sum().item()

		conf_matrices[f'class_{cls}']['TP'] += TP
		conf_matrices[f'class_{cls}']['TN'] += TN
		conf_matrices[f'class_{cls}']['FP'] += FP
		conf_matrices[f'class_{cls}']['FN'] += FN


def calculate_metrics(metrics, conf_matrices, num_classes):
	TP, TN, FP, FN = 0, 0, 0, 0

	for cls in range(0, num_classes):
		TP += conf_matrices[f'class_{cls}']['TP']
		TN += conf_matrices[f'class_{cls}']['TN']
		FP += conf_matrices[f'class_{cls}']['FP']
		FN += conf_matrices[f'class_{cls}']['FN']
	
	metrics['accuracy'] = accuracy(TP, TN, FP, FN)
	metrics['precision'] = precision(TP, TN, FP, FN)
	metrics['recall'] = recall(TP, TN, FP, FN)
	metrics['f1'] = f1(TP, TN, FP, FN)
	metrics['iou'] = iou(TP, TN, FP, FN)
	metrics['dice'] = dice(TP, TN, FP, FN)
	metrics['conf_matrices'] = conf_matrices

def accuracy(TP, TN, FP, FN):
	return (TP + TN) / (TP + TN + FP + FN + 1e-6)
				
def precision(TP, TN, FP, FN):
	return TP / (TP + FP + 1e-6)

def	recall(TP, TN, FP, FN):
	return TP / (TP + FN + 1e-6)

def f1(TP, TN, FP, FN):
	return 2 * (precision(TP, TN, FP, FN) * recall(TP, TN, FP, FN)) / (precision(TP, TN, FP, FN) + recall(TP, TN, FP, FN) + 1e-6)

def iou(TP, TN, FP, FN):
	return TP / (TP + FP + FN + 1e-6)

def dice(TP, TN, FP, FN):
	return 2 * TP / (2 * TP + FP + FN + 1e-6)