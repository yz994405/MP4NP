import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import csv
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from src.utils.metrics import calculate_metrics

class MultiModalTrainer:
    def __init__(self, model, device, lr_rate=0.0001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.min_lr = 0.01e-5   
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'val_r2': [],
            'val_rm2': [],
            'val_pearson': [], 
            'val_spearman': [],  
            'val_mse': [],
            'val_ci': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        self.result_file = None

    def set_result_file(self, result_file):
        self.result_file = result_file
        if self.result_file:
            with open(self.result_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'RMSE', 'MSE', 'MAE', 'R2', 'RM2', 'Pearson', 'Spearman', 'CI'])

    def record_metrics_to_csv(self, epoch, train_loss, val_loss, rmse, mse, mae, r2, rm2, pearson, spearman, ci):
        if self.result_file:
            with open(self.result_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, rmse, mse, mae, r2, rm2, pearson, spearman, ci])

    def record_best_metrics_to_csv(self):
        if self.result_file and len(self.history['train_loss']) > 0:
            with open(self.result_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([''] * 11)
                best_train_loss = min(self.history['train_loss'])
                best_val_loss = min(self.history['val_loss'])
                best_rmse = min(self.history['val_rmse'])
                best_mse = min(self.history['val_mse'])
                best_mae = min(self.history['val_mae'])
                best_r2 = max(self.history['val_r2'])
                best_rm2 = max(self.history['val_rm2'])
                best_pearson = max(self.history['val_pearson'])
                best_spearman = max(self.history['val_spearman'])
                best_ci = max(self.history['val_ci'])
                writer.writerow([
                    'Best Values', 
                    f'{best_train_loss:.6f}', 
                    f'{best_val_loss:.6f}', 
                    f'{best_rmse:.6f}', 
                    f'{best_mse:.6f}', 
                    f'{best_mae:.6f}', 
                    f'{best_r2:.6f}', 
                    f'{best_rm2:.6f}',
                    f'{best_pearson:.6f}', 
                    f'{best_spearman:.6f}', 
                    f'{best_ci:.6f}'
                ])

    # Training an epoch
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training', leave=False, ncols=100)
        
        for batch_idx, (chembert_x, grover_x, graphmvp_x, protein_x, affinity) in enumerate(pbar):
            chembert_x, grover_x, graphmvp_x, protein_x, affinity = chembert_x.to(self.device), grover_x.to(self.device), graphmvp_x.to(self.device), protein_x.to(self.device), affinity.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(chembert_x, grover_x, graphmvp_x, protein_x)
            loss = self.criterion(output.squeeze(), affinity)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * affinity.size(0)
            total_samples += affinity.size(0)
            current_avg_loss = total_loss / total_samples
            pbar.set_postfix({'Loss': f'{current_avg_loss:.6f}'})
        avg_loss = total_loss / total_samples
        self.scheduler.step()  
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < self.min_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.min_lr
            print(f"Learning rate has reached the minimum value {self.min_lr}, no longer decaying")
            
        return {'train_loss': avg_loss}

    # Val an epoch 
    def valid_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_affinities = []
        
        pbar = tqdm(val_loader, desc='Val', leave=False, ncols=100)
        
        with torch.no_grad():
            for chembert_x, grover_x, graphmvp_x, protein_x, affinity in pbar:
                chembert_x, grover_x, graphmvp_x, protein_x, affinity = chembert_x.to(self.device), grover_x.to(self.device), graphmvp_x.to(self.device), protein_x.to(self.device), affinity.to(self.device)
                output = self.model(chembert_x, grover_x, graphmvp_x, protein_x)
                loss = self.criterion(output.squeeze(), affinity)
                total_loss += loss.item() * affinity.size(0)
                all_predictions.extend(output.squeeze().cpu().numpy())
                all_affinities.extend(affinity.cpu().numpy())
                current_avg_loss = total_loss / len(all_affinities)
                pbar.set_postfix({'Val Loss': f'{current_avg_loss:.6f}'})
        avg_loss = total_loss / len(val_loader.dataset)
        metrics = calculate_metrics(np.array(all_affinities), np.array(all_predictions))
        metrics['val_loss'] = avg_loss
        return metrics
    
    # Train
    def train(self, train_loader, val_loader, num_epochs, patience, use_early_stopping, save_dir, result_file=None):
        if result_file:
            self.set_result_file(result_file)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.valid_epoch(val_loader)
            
            if self.result_file:
                self.record_metrics_to_csv(
                    epoch=epoch,
                    train_loss=train_metrics['train_loss'],
                    val_loss=val_metrics['val_loss'],
                    rmse=val_metrics['rmse'],
                    mse=val_metrics['mse'],
                    mae=val_metrics['mae'],
                    r2=val_metrics['r2'],
                    rm2=val_metrics['rm2'],
                    pearson=val_metrics['pearson'],
                    spearman=val_metrics['spearman'],
                    ci=val_metrics['ci']
                )
            
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['val_pearson'].append(val_metrics['pearson'])  
            self.history['val_spearman'].append(val_metrics['spearman'])  
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_ci'].append(val_metrics['ci'])
            self.history['val_rm2'].append(val_metrics['rm2'])
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{num_epochs}:")
                print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
                print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
                print(f"  Val MAE: {val_metrics['mae']:.6f}")
                print(f"  Val R²: {val_metrics['r2']:.6f}")
                print(f"  Val RM²: {val_metrics['rm2']:.6f}")
                print(f"  Val Pearson: {val_metrics['pearson']:.6f}")  
                print(f"  Val Spearman: {val_metrics['spearman']:.6f}")  
                print(f"  Val MSE: {val_metrics['mse']:.6f}")
                print(f"  Val CI: {val_metrics['ci']:.6f}")
                print(f"  Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.6g}")
            
            # Best model saving
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                best_model_path = save_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), str(best_model_path))
                print(f"Best model saved (Epoch {epoch})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if use_early_stopping and self.patience_counter >= patience:
                print(f"Early stopping triggered! {patience} epochs without improvement")
                break

        # Write best metrics to CSV file
        if self.result_file:
            self.record_best_metrics_to_csv()
            print(f"Best metrics recorded in: {self.result_file}")
        
        print(f"Training completed! Best model at Epoch {self.best_epoch}")
        return self.history