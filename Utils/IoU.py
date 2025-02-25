import torch
import math
import numpy as np

def ciou(box1, box2, eps=1e-7): 
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU      #IoU    
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex  width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    fin_iou =  iou - (rho2 / c2 + v * alpha)  
    return fin_iou.mean()

def inner_ciou(box1, box2, ratio = 1.0, xywh=True, eps=1e-7): 
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU      #IoU    
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
   #Inner-IoU      #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU       
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio,\
                                                             y1 - h1_*ratio, y1 + h1_*ratio
    inner_b2_x1,inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio,\
                                                             y2 - h2_*ratio, y2 + h2_*ratio
    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                   (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    inner_union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps
    inner_iou = inner_inter/inner_union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex  width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    final_iou =  inner_iou - (rho2 / c2 + v * alpha)  
    return final_iou.mean()

def n_ciou(box1, box2, n=2, ratio=1.0, eps=1e-7):

    (x1, y1, w1, h1), (x2, y2, w2, h2) = torch.chunk(box1, 4, dim=-1), torch.chunk(box2, 4, dim=-1)

    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2

    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    # print('iou', iou)
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio, y1 - h1_*ratio, y1 + h1_*ratio
    inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio, y2 - h2_*ratio, y2 + h2_*ratio

    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                   (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)

    inner_union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps

    n_iou = (inner_inter + n * inner_inter) / (inner_union + n * inner_inter)
    # print('n_iou', n_iou)
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    # print('v', v)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
        # print('al', alpha)

    fin_iou = n_iou - (rho2 / c2 + v * alpha)
    result = fin_iou.detach().cpu().numpy()
    result = result.mean()
    if np.isnan(result):
        result = 0
    # print('fin', fin_iou)
    # return fin_iou.mean(dim=(1, 2, 3, 4))
    return result
# box1 = torch.rand(2, 2, 128, 128, 32)
# box2 = torch.rand(2, 2, 128, 128, 32)
# print(box1.shape)

# mean_iou = n_ciou(box1, box2, n=5)
# print(mean_iou.mean())            # Output a scalar value representing the average n-CIoU for all bounding boxes