import numpy as np


class Metrics():
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.conf_matrix =np.ndarray((num_classes, num_classes), dtype=np.int64)
		self.metrics = { }
		self.reset()

	def reset(self):
		self.conf_matrix.fill(0)


	def add(self, logits, target):
		_, predicted = logits.max(1)

		predicted = predicted.view(-1).cpu().numpy()
		target = target.view(-1).cpu().numpy()

		x = predicted + self.num_classes * target
		bincount_2d = np.bincount(
			x.astype(np.int64), minlength=self.num_classes**2)
		# assert bincount_2d.size == self.num_classes**2
		conf = bincount_2d.reshape((self.num_classes, self.num_classes))

		self.conf_matrix += conf

	
	def _get_conf_matrix_values(self):
		tp = np.diag(self.conf_matrix)
		fp = np.sum(self.conf_matrix, 0) - tp
		fn = np.sum(self.conf_matrix, 1) - tp
		tn = np.sum(self.conf_matrix) - tp - fp - fn

		return tp, tn, fp, fn
	

	def get_metrics(self):
		
		tp, tn, fp, fn =self._get_conf_matrix_values()

		with np.errstate(divide='ignore', invalid='ignore'):
			metrics = {
				'epoch': 0,
				'train_loss': 0,
				'val_loss'
				'confusion_matrix': self.conf_matrix,
				'accuracy': (tp + tn) / (tp + tn + fp + fn),
				'precision': tp / (tp + fp),
				'recall': tp / (tp + fn),
				'f1': tp / (tp + (fp + fn) / 2.),
				'dice': 2 * tp / (2 * tp + fp + fn),
				'iou': tp / (tp + fp + fn)
			}

		return metrics
	
	def get_iou(self):
		
		tp, _, fp, fn =self._get_conf_matrix_values()
		with np.errstate(divide='ignore', invalid='ignore'):
			iou = tp / (tp + fp + fn)
		return iou