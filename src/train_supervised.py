import time
import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, train_loader, criterion, optimizer, metrics, device):
	running_loss = 0.
	metrics.reset()

	for images, masks in tqdm(train_loader, desc=f"Training"):
		images = images.to(device) # [N, C, H, W]
		target = masks.long().to(device)  # [N, H, W]
		
		# Forward propagation
		logits = model(images) # [N, C, H, W]

		# Loss computation
		loss = criterion(logits, target)

		# Back propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Add loss to epoch
		running_loss += loss.item()

		# Add metrics to epoch
		metrics.add(logits.detach(), target.detach())
	
	epoch_metrics = metrics.get_metrics()
	return running_loss / len(train_loader), epoch_metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, metrics, save_folder, experiment, device):
	checkpoints_folder = save_folder / 'checkpoints' / f"{experiment}"
	checkpoints_folder.mkdir(exist_ok=True)
	logs_folder = save_folder / 'logs' / f"{experiment}"
	logs_folder.mkdir(exist_ok=True)

	# To save the best model
	training_metrics = []
	best_iou = 0.

	writer = SummaryWriter(logs_folder)

	start_training = time.time()
	for epoch in range(epochs):
		print(f"📘 Epoch [{epoch+1}/{epochs}]")
		
		# Training
		model.train()
		avg_train_loss, epoch_metrics = train_one_epoch(model, train_loader, criterion, optimizer, metrics, device)

		print(f"   🟢 Train Loss: {avg_train_loss:.4f} -- Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}")
		print(f"   🔵 IoU       : {np.nanmean(epoch_metrics['iou']):4f}")

		# Validation	
		val_loss = 0.
		metrics.reset()
		model = model.eval()
		with torch.no_grad():
			for images, masks in tqdm(val_loader, desc=f"Validation"):
				images = images.to(device)
				target = masks.long().to(device)  # (B, H, W)
		
				# forward
				logits = model(images)  # (B, C, H, W)
				loss = criterion(logits, target)
				val_loss += loss.item()
				metrics.add(logits.detach(), target.detach())
		
		avg_val_loss = val_loss / len(val_loader)
		epoch_metrics = metrics.get_metrics()
		epoch_metrics['epoch'] = epoch + 1
		epoch_metrics['train_loss'] = avg_train_loss
		epoch_metrics['val_loss'] = avg_train_loss
		val_iou = np.nanmean(epoch_metrics['iou'])
		training_metrics.append(epoch_metrics)
		# writer.add_scalar('Loss/training', avg_train_loss, epoch + 1)
		# writer.add_scalar("Loss/validation", avg_val_loss, epoch + 1)
		writer.add_scalars('Loss', {'training':avg_train_loss,
									'validation':avg_val_loss}, epoch + 1)
		writer.add_scalar("IoU", val_iou, epoch + 1)


		print(f"   🔵 Val Loss : {avg_val_loss:.4f} -- Elapsed: {datetime.timedelta(seconds=time.time()-start_training)}")
		print(f"   🔵 accuracy : {np.nanmean(epoch_metrics['accuracy']):4f}. Class: {epoch_metrics['accuracy']:4f}")
		print(f"   🔵 precision: {np.nanmean(epoch_metrics['precision']):4f}. Class: {epoch_metrics['precision']:4f}")
		print(f"   🔵 recall   : {np.nanmean(epoch_metrics['recall']):4f}. Class: {epoch_metrics['recall']:4f}")
		print(f"   🔵 f1       : {np.nanmean(epoch_metrics['f1']):4f}. Class: {epoch_metrics['f1']:4f}")
		print(f"   🔵 IoU      : {val_iou:4f}. Class: {epoch_metrics['iou']:4f}")
		print(f"   🔵 Dice     : {np.nanmean(epoch_metrics['dice']):4f}. Class: {epoch_metrics['dice']:4f}")

				
		# Save best model
		if val_iou > best_iou:
			best_iou = val_iou
			
			best_model_file = checkpoints_folder / 'best_model.pth'
			torch.save(model.state_dict(), best_model_file)
			print("💾 Best model saved!")
		print("........................................................................")

	metrics_file = checkpoints_folder / 'metrics.pt'
	torch.save(training_metrics, metrics_file)
	print("🏁 Entrenament complet.")
	writer.close()