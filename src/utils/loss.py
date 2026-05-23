import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 3.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 1D Tensor of class weights. Shape: (num_classes,)
            gamma: Focusing parameter.
            reduction: 'mean', 'sum', or 'none'
        """
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Get standard cross entropy loss (unweighted)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Get the probability of the true class
        pt = torch.exp(-ce_loss)
        
        # 3. Calculate the standard focal modulating term
        focal_term = (1 - pt)**self.gamma


        # 4. Apply class-specific alpha if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as the targets (e.g., GPU)
            self.alpha = self.alpha.to(targets.device)
            
            # Gather the alpha weight for the specific target class of each sample
            # targets shape: (Batch,), alpha shape: (Classes,) -> alpha_t shape: (Batch,)
            alpha_t = self.alpha.gather(0, targets)
            
            # Multiply everything together
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        # 5. Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss