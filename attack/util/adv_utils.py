import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitsAdvLoss(nn.Module):

    def __init__(self, kappa=0.):
        """Adversarial function on logits.

        Args:
            kappa (float, optional): min margin. Defaults to 0..
        """
        super(LogitsAdvLoss, self).__init__()

        self.kappa = kappa

    def forward(self, logits, targets):
        """Adversarial loss function using logits.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        B, K = logits.shape
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()
        one_hot_targets = torch.zeros(B, K).cuda().scatter_(
            1, targets, 1).float()  # to one-hot
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max((1. - one_hot_targets) * logits -
                                 one_hot_targets * 10000., dim=1)[0]
        loss = torch.clamp(other_logits - real_logits + self.kappa, min=0.)
        return loss.mean()


class CrossEntropyAdvLoss(nn.Module):

    def __init__(self):
        """Adversarial function on output probabilities.
        """
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, targets):
        """Adversarial loss function using cross entropy.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        loss = F.cross_entropy(logits, targets)
        return loss


class PoseEntropyAdvLoss(nn.Module):

    def __init__(self):
        """Adversarial function on output probabilities.
        """
        super(PoseEntropyAdvLoss, self).__init__()

    def forward(self, pre, gt):
        """Adversarial loss function using cross entropy.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.FloatTensor): attack target class
        """
        loss = 1/torch.mean((pre - gt)**2)
        # loss = 1/torch.exp(torch.mean((pre - gt)**2))
        return loss
