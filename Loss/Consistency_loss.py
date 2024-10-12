import torch
from torch.nn import functional as F
import numpy as np

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    # print(input_logits.size(),target_logits.size())
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    loss = torch.nn.MSELoss(reduction='sum') 
    mse_loss = loss(input_softmax, target_softmax)

    return mse_loss

# def softmax_mse_loss(inputs, targets,
#                      conf_mask=False, threshold=None,
#                      use_softmax=False):

#     assert inputs.requires_grad is True and targets.requires_grad is False
#     assert inputs.size() == targets.size()  # (batch_size * num_classes * H * W)
#     inputs = F.softmax(inputs, dim=1)
#     if use_softmax:
#         targets = F.softmax(targets, dim=1)

#     if conf_mask:
#         # method 1):
#         mask = (targets.max(1)[0] > threshold)
#         if not torch.any(mask):
#             loss_mat = F.mse_loss(inputs, targets, reduction='none') * .0
#         else:
#             loss_mat = F.mse_loss(inputs, targets, reduction='none')
#             loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]

#         return loss_mat.mean()
#         # method 2).
#         # mask = (targets.max(1)[0] > threshold)
#         # loss_mat = F.mse_loss(inputs, targets, reduction='none')
#         # loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
#         # if loss_mat.shape.numel() == 0:
#         #     print(dist.get_rank())
#         #     loss_mat = F.mse_loss(inputs, targets, reduction='none') * .0
#         # return loss_mat.mean()
#     return F.mse_loss(inputs, targets, reduction='mean')  # take the mean over the batch_size



def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)