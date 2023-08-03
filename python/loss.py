# From CenterNet: https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/losses.py
import torch
import torch.nn as nn


def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """
    eps = 1e-16
    # print(torch.min(pred))
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    gamma = 1
    # alpha = 0.75

    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = (
        torch.log(1 - pred + eps) * torch.pow(pred, gamma) * neg_weights * neg_inds
    )

    # print(pos_loss)
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    # print(pos_loss)
    # print(neg_loss)

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    # print(loss)
    return loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)
