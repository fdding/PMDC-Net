import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, ignore_index=None, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
       
        inputs = F.softmax(inputs, dim=1)
        
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            targets = targets * mask
        
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        for i in range(num_classes):
            input_flat = inputs[:, i].reshape(-1)
            target_flat = targets_one_hot[:, i].reshape(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (
                input_flat.sum() + target_flat.sum() + self.smooth
            )
            dice_scores.append(1.0 - dice)  
        
        dice_loss = torch.stack(dice_scores)
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class WeightedCombinedLoss(nn.Module):

    def __init__(self, ce_weight=0.3, dice_weight=0.7, smooth=1.0, ignore_index=None):
        super(WeightedCombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
        if ignore_index is not None:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)

    def forward(self, inputs, targets):
      
        ce_loss = self.ce_loss(inputs, targets)
        
        dice_loss = self.dice_loss(inputs, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        losses = {
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'total_loss': total_loss.item(),
            'weights': {
                'ce': self.ce_weight,
                'dice': self.dice_weight
            }
        }
        
        return total_loss, losses

def get_loss_function(**kwargs):

    return WeightedCombinedLoss(**kwargs)


