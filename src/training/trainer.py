import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Optional

class MultiOmicsTrainer:
    """
    Handles the training, validation, checkpointing, and visualization 
    of the Patient-Specific Multi-Omics Graph Transformer.
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 dynamic_pe_module: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 save_dir: str = "checkpoints",
                 viz_dir: str = "viz",
                 use_dropedge: bool = True): 
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dynamic_pe = dynamic_pe_module.to(device)
        self.device = device
        self.use_dropedge = use_dropedge
        
        # Directories
        self.save_dir = save_dir
        self.viz_dir = viz_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.best_val_f1 = 0.0
        
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'val_acc': [], 'val_f1': [], 'val_auc': []
        }

        # CACHING THE STATIC GRAPH
        # We pre-compute the pristine, un-dropped graph PE once.
        # This is used for validation, and for training if DropEdge is turned off.
        print("Pre-computing static Graph PE for validation baseline...")
        self.dynamic_pe.eval() # Temporarily disable dropout behavior
        with torch.no_grad():
            self.static_pe = self.dynamic_pe()
        self.dynamic_pe.train() # Set back to train mode

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        # Determine which Graph PE to use for this epoch
        if self.use_dropedge:
            # Generate a fresh, perturbed structural topology
            current_graph_pe = self.dynamic_pe()
        else:
            # Reuse the static topology (saves computation time)
            current_graph_pe = self.static_pe
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            rna = batch['mRNA'].to(self.device)
            cnv = batch['CNV'].to(self.device)
            methy = batch['methy'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(rna, cnv, methy, current_graph_pe)
            logits = outputs['logits']
            
            loss = self.criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in self.val_loader:
            rna = batch['mRNA'].to(self.device)
            cnv = batch['CNV'].to(self.device)
            methy = batch['methy'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # ALWAYS evaluate on the pristine, un-dropped static graph
            outputs = self.model(rna, cnv, methy, self.static_pe)
            logits = outputs['logits']
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.0
        
        return {"val_loss": avg_loss, "val_acc": acc, "val_f1": f1, "val_auc": auc}



    def fit(self, epochs: int):
        print(f"Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()
            
            # Append to history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_acc'].append(val_metrics['val_acc'])
            self.history['val_f1'].append(val_metrics['val_f1'])
            self.history['val_auc'].append(val_metrics['val_auc'])
            
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val F1: {val_metrics['val_f1']:.4f} | "
                  f"Val AUC: {val_metrics['val_auc']:.4f} | "
                  f"Val ACC: {val_metrics['val_acc']:.4f}")
            
            if val_metrics['val_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['val_f1']
                self._save_checkpoint("best_model.pth")
                print(">>> New best model saved (Macro F1)!")
                
        # Generate and save the visualization at the end of training
        self._plot_metrics()

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1
        }, path)

    def _plot_metrics(self):
        """Generates a 4-panel plot of training and validation metrics."""
        print(f"\nGenerating training visualization in {self.viz_dir}/training_history.png ...")
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=4)
        plt.plot(epochs, self.history['val_loss'], label='Val Loss', color='red', marker='o', markersize=4)
        plt.title('Cross Entropy Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: Macro F1
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history['val_f1'], label='Val Macro F1', color='purple', marker='o', markersize=4)
        plt.title('Macro F1-Score (Primary Metric)')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 3: Macro AUC
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history['val_auc'], label='Val Macro AUC', color='green', marker='o', markersize=4)
        plt.title('Macro AUC (One-vs-Rest)')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 4: Accuracy
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.history['val_acc'], label='Val Accuracy', color='orange', marker='o', markersize=4)
        plt.title('Overall Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()