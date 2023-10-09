import torch
import torch.nn as nn
import torch.nn.functional as F


def hm_focal_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, gt, pos):
    batch, dim, height, width = gt.size()
    # Computes input>other element-wise.
    pos = pos.gt(0)
    obj_num = pos.sum()
    if obj_num == 0:
        return torch.Tensor([0.0]).to(pred.device)
    pos_inds = pos.expand(batch, dim, height, width)

    x = pred[pos_inds]
    gt_x = gt[pos_inds]
    x_loss = nn.functional.l1_loss(x, gt_x, size_average=False)
    loss = x_loss / (obj_num + 1e-4)

    return loss


def hps_l1_loss(pred, gt, hps_pos):
    # Computes input>other element-wise.
    hps_pos = hps_pos.gt(0)
    obj_num = hps_pos.sum()
    if obj_num == 0:
        return torch.tensor(0.0).to(pred.device)

    x = pred[hps_pos]
    gt_x = gt[hps_pos]
    x_loss = nn.functional.l1_loss(x, gt_x, size_average=False)
    loss = x_loss / (obj_num + 1e-4)

    return loss


def hps_conf_ce_loss(pred, hps_vis_pos, hps_unvis_pos):
    hps_vis_pos_bool = hps_vis_pos.gt(0)
    hps_vis_obj_num = hps_vis_pos_bool.sum()
    if hps_vis_obj_num == 0:
        hps_vis_loss = torch.tensor(0.0).to(pred.device)
    else:
        hps_vis_loss = F.binary_cross_entropy(F.sigmoid(pred[hps_vis_pos_bool]), hps_vis_pos[hps_vis_pos_bool])
    hps_unvis_pos_bool = hps_unvis_pos.gt(0)
    hps_unvis_obj_num = hps_unvis_pos_bool.sum()
    if hps_unvis_obj_num == 0:
        hps_unvis_loss = torch.tensor(0.0).to(pred.device)
    else:
        hps_unvis_loss = F.binary_cross_entropy(F.sigmoid(pred[hps_unvis_pos_bool]), hps_unvis_pos[hps_unvis_pos_bool] * 0)
    loss = hps_vis_loss + hps_unvis_loss
    return loss