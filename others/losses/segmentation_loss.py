import torch
import torch.nn as nn
import torch.nn.functional as F

from others.losses import CustomLoss


def dice_coff(output, target, smooth, g_dice, eps):
    ns = output.size()[1]
    if ns >= 2:
        output = torch.softmax(output, dim=1)
        target_onehot = torch.zeros_like(output, device=output.device, dtype=torch.float32)
        with torch.no_grad():
            target_onehot.scatter_(1, target.long().unsqueeze(1), 1)
    else:
        with torch.no_grad():
            target_onehot = target.unsqueeze(1)
        output = torch.sigmoid(output)
    target_onehot = target_onehot.flatten(2)
    output = output.flatten(2)
    w = 1
    if g_dice:
        w = torch.sum(target_onehot, dim=-1)
        w = 1 / (w ** 2 + eps)
    inter = w * torch.sum(output * target_onehot, dim=-1)
    union = w * torch.sum(output + target_onehot, dim=-1)
    _coff = (2 * inter + smooth) / (union + smooth)
    return _coff


class DiceLoss(CustomLoss):
    """ Dice loss for the segmentation task """
    def __init__(self, smooth_factor=1.0, use_generalized_dice=False, eps=1e-9):
        """
        :param smooth_factor: smooth value to avoid division by zero
        :param use_generalized_dice: use generalized dice loss
        :param eps: epsilon value to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth_factor = smooth_factor
        self.use_generalized_dice = use_generalized_dice
        self.eps = eps

    def forward(self, output, target):
        coff = dice_coff(output, target, smooth=self.smooth_factor, g_dice=self.use_generalized_dice, eps=self.eps)
        loss = 1 - coff
        return loss.mean()


class FocalLoss(nn.Module):
    """ Focal loss for the segmentation task """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        :param alpha: Weighting factor for each class. Can be a list of weights for each class or a single float for binary classification.
        :param gamma: Focusing parameter to adjust the rate at which easy examples are down-weighted.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1 - alpha])
            else:
                self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: Predictions from the model (logits), shape (batch_size, num_classes).
        :param targets: Ground truth labels, shape (batch_size) for multi-class or (batch_size, 1) for binary classification.
        """
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(-1))

        targets = targets.view(-1, 1)

        log_pt = F.log_softmax(inputs, dim=-1)
        log_pt = log_pt.gather(1, targets)
        log_pt = log_pt.view(-1)
        pt = log_pt.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.view(-1))
            log_pt = log_pt * at

        loss = -1 * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss