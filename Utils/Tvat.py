import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    PS-MT
    [CVPR'22] Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation
    by Yuyuan Liu, Yu Tian, Yuanhong Chen, Fengbei Liu, Vasileios Belagiannis and Gustavo Carneiro
    copyright (c) Computer Vision and Pattern Recognition Conference (CVPR), 2022

'''

'''
    copyright (c) https://github.com/yyliu01/PS-MT/blob/main/CityCode/Model/Deeplabv3_plus/encoder_decoder.py
'''
def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv_t(x, ema_model1, ema_model2, it=1, xi=1e-1, eps=10.0):

    # stop bn
    ema_model1.eval()
    ema_model2.eval()
    
    x_detached = x
    # print('x_detached',x_detached.shape)

    with torch.no_grad():
        pred = F.softmax((ema_model1(x_detached)[0] + ema_model2(x_detached)[0])/2, dim=1)

    d = torch.rand(x.shape).sub(0.2).to(x.device)    # noise float [-0.2, 0.2]
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        d.requires_grad_()
        pred_hat = (ema_model1(x_detached + xi * d)[0] + ema_model2(x_detached + xi * d)[0])/2
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        # model1.zero_grad()
        # model2.zero_grad()

    r_adv = d * eps

    # reopen bn, but freeze other params.
    # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16
    # model1.train()
    # model2.train()
    return r_adv[0:2]        # Only the noise of the original picture is retained, exactly is the first two elements of dimension 0

'''
    copyright (c) https://github.com/yyliu01/PS-MT/blob/main/CityCode/Utils/losses.py
'''
def semi_ce_loss(inputs, targets,
                 conf_mask=True, threshold=None,
                 threshold_neg=None, temperature_value=1):
    # target => logit, input => logit
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets/temperature_value, dim=1)
        # print('input',inputs.shape,'target',targets.shape)
        # for positive
        targets_real_prob = F.softmax(targets, dim=1)
        
        weight = targets_real_prob.max(1)[0]
        total_number = len(targets_prob.flatten(0))
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)

        mask_neg = (targets_prob < threshold_neg)

        # temp negative label * mask_neg, which mask down the positive labels.
        # neg_label = torch.ones(targets.shape, dtype=targets.dtype, device=targets.device) * mask_neg
        neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype)
        # if neg_label.shape[-1] != 19:
        #     neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
        #                                                    neg_label.shape[2], neg_label.shape[3], neg_label.shape[-1]]).cuda()),
        #                           dim=0)
        # print('target_prob',targets_prob.shape, 'neg_label',neg_label.shape)
        neg_label = neg_label.permute(0, 4, 1, 2, 3)
        neg_label = 1 - neg_label
        # print('neg_label_',neg_label.shape)
        if not torch.any(mask):
            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            # zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            return inputs.sum() * .0, pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight

            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            # print('neg_prediction_prob',neg_prediction_prob.shape)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))

            return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
    else:
        raise NotImplementedError