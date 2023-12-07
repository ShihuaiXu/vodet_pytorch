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
    # Computes input > other element-wise.
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
    # Computes input > other element-wise.
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
    select_index = [0, 2, 4, 6]
    hps_vis_pos = hps_vis_pos[:, select_index, :, :]
    hps_unvis_pos = hps_unvis_pos[:, select_index, :, :]
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


def hps_cyc3_loss(pred_coord, pred_conf, gt, hps_pos, hps_vis_pos, hps_unvis_pos):
    hps_coord_pos_bool = hps_pos.gt(0)
    hps_conf_pos_bool = hps_pos[:, [0, 2, 4], :, :].gt(0)
    hps_vis_pos_bool = hps_vis_pos.gt(0)
    hps_unvis_pos_bool = hps_unvis_pos.gt(0)
    vis_obj_num = hps_vis_pos_bool.sum()
    unvis_obj_num = hps_unvis_pos_bool.sum()
    if vis_obj_num == 0 and unvis_obj_num == 0:
        loss_coord = torch.tensor(0.0).to(pred_coord.device)
        loss_vis_conf = torch.tensor(0.0).to(pred_coord.device)
        loss_unvis_conf = torch.tensor(0.0).to(pred_coord.device)
    elif vis_obj_num != 0 and unvis_obj_num == 0:
        loss_unvis_conf = torch.tensor(0.0).to(pred_coord.device)
        ##############################
        # kps coord loss
        gt_x1 = gt[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        gt_y1 = gt[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        gt_x2 = gt[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        gt_y2 = gt[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        gt_x3 = gt[:, 4, :, :][hps_coord_pos_bool[:, 4, :, :]].unsqueeze(-1)
        gt_y3 = gt[:, 5, :, :][hps_coord_pos_bool[:, 5, :, :]].unsqueeze(-1)

        gt_x1y1 = torch.cat([gt_x1, gt_y1], axis=1)
        gt_x2y2 = torch.cat([gt_x2, gt_y2], axis=1)
        gt_x3y3 = torch.cat([gt_x3, gt_y3], axis=1)

        gt_xy_1 = torch.cat([gt_x1y1, gt_x2y2, gt_x3y3], axis=1)
        gt_xy_2 = torch.cat([gt_x2y2, gt_x3y3, gt_x1y1], axis=1)
        gt_xy_3 = torch.cat([gt_x3y3, gt_x1y1, gt_x2y2], axis=1)

        mask_x1 = hps_vis_pos[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        mask_y1 = hps_vis_pos[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        mask_x2 = hps_vis_pos[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        mask_y2 = hps_vis_pos[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        mask_x3 = hps_vis_pos[:, 4, :, :][hps_coord_pos_bool[:, 4, :, :]].unsqueeze(-1)
        mask_y3 = hps_vis_pos[:, 5, :, :][hps_coord_pos_bool[:, 5, :, :]].unsqueeze(-1)

        mask_x1y1 = torch.cat([mask_x1, mask_y1], axis=1)
        mask_x2y2 = torch.cat([mask_x2, mask_y2], axis=1)
        mask_x3y3 = torch.cat([mask_x3, mask_y3], axis=1)

        mask_xy_1 = torch.cat([mask_x1y1, mask_x2y2, mask_x3y3], axis=1)
        mask_xy_2 = torch.cat([mask_x2y2, mask_x3y3, mask_x1y1], axis=1)
        mask_xy_3 = torch.cat([mask_x3y3, mask_x1y1, mask_x2y2], axis=1)

        dt_x1 = pred_coord[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_y1 = pred_coord[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_x2 = pred_coord[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        dt_y2 = pred_coord[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        dt_x3 = pred_coord[:, 4, :, :][hps_coord_pos_bool[:, 4, :, :]].unsqueeze(-1)
        dt_y3 = pred_coord[:, 5, :, :][hps_coord_pos_bool[:, 5, :, :]].unsqueeze(-1)
        dt_xy = torch.cat([dt_x1, dt_y1, dt_x2, dt_y2, dt_x3, dt_y3], axis=1)
        dt_xy_1 = dt_xy * mask_xy_1
        dt_xy_2 = dt_xy * mask_xy_2
        dt_xy_3 = dt_xy * mask_xy_3

        loss1 = torch.abs(dt_xy_1 - gt_xy_1)
        loss2 = torch.abs(dt_xy_2 - gt_xy_2)
        loss3 = torch.abs(dt_xy_3 - gt_xy_3)

        gt_xy_1 = gt_xy_1.unsqueeze(1)
        gt_xy_2 = gt_xy_2.unsqueeze(1)
        gt_xy_3 = gt_xy_3.unsqueeze(1)
        gt_xy_cat = torch.cat([gt_xy_1, gt_xy_2, gt_xy_3], axis=1)

        dt_xy_1 = dt_xy_1.unsqueeze(1)
        dt_xy_2 = dt_xy_2.unsqueeze(1)
        dt_xy_3 = dt_xy_3.unsqueeze(1)
        dt_xy_cat = torch.cat([dt_xy_1, dt_xy_2, dt_xy_3], axis=1)

        loss1_sum = torch.sum(loss1, dim=-1).unsqueeze(-1)
        loss2_sum = torch.sum(loss2, dim=-1).unsqueeze(-1)
        loss3_sum = torch.sum(loss3, dim=-1).unsqueeze(-1)
        loss_sum_cat = torch.cat([loss1_sum, loss2_sum, loss3_sum], axis=-1)

        first_pt_gt = torch.argmin(loss_sum_cat, axis=-1)
        first_pt_gt_onehot = F.one_hot(first_pt_gt, num_classes=3)

        pred_coord_pt_gt = dt_xy_cat[first_pt_gt_onehot == 1]
        target_coord_pt_gt = gt_xy_cat[first_pt_gt_onehot == 1]
        loss_coord = F.l1_loss(pred_coord_pt_gt, target_coord_pt_gt, size_average=False)
        loss_coord = loss_coord / (mask_xy_1.sum() + 1e-4)
        ##############################
        # kps vis conf loss
        gt_vis_1 = torch.cat([mask_x1, mask_x2, mask_x3], axis=1)
        gt_vis_2 = torch.cat([mask_x2, mask_x3, mask_x1], axis=1)
        gt_vis_3 = torch.cat([mask_x3, mask_x1, mask_x2], axis=1)

        dt_conf_1 = pred_conf[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_conf_2 = pred_conf[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_conf_3 = pred_conf[:, 2, :, :][hps_conf_pos_bool[:, 2, :, :]].unsqueeze(-1)
        dt_conf = torch.cat([dt_conf_1, dt_conf_2, dt_conf_3], axis=1)
        dt_vis_1 = F.sigmoid(dt_conf) * gt_vis_1
        dt_vis_2 = F.sigmoid(dt_conf) * gt_vis_2
        dt_vis_3 = F.sigmoid(dt_conf) * gt_vis_3

        gt_vis_1 = gt_vis_1.unsqueeze(1)
        gt_vis_2 = gt_vis_2.unsqueeze(1)
        gt_vis_3 = gt_vis_3.unsqueeze(1)
        gt_vis_ts_cat = torch.cat([gt_vis_1, gt_vis_2, gt_vis_3], axis=1)

        dt_vis_1 = dt_vis_1.unsqueeze(1)
        dt_vis_2 = dt_vis_2.unsqueeze(1)
        dt_vis_3 = dt_vis_3.unsqueeze(1)
        dt_vis_cat = torch.cat([dt_vis_1, dt_vis_2, dt_vis_3], axis=1)

        pred_vis_pt_gt = dt_vis_cat[first_pt_gt_onehot == 1]
        target_vis_pt_gt = gt_vis_ts_cat[first_pt_gt_onehot == 1]
        loss_vis_conf = F.binary_cross_entropy(pred_vis_pt_gt, target_vis_pt_gt)
        ##############################
    elif vis_obj_num == 0 and unvis_obj_num != 0:
        loss_coord = torch.tensor(0.0).to(pred_coord.device)
        loss_vis_conf = torch.tensor(0.0).to(pred_coord.device)
        ##############################
        # kps unvis conf loss
        dt_conf_1 = pred_conf[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_conf_2 = pred_conf[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_conf_3 = pred_conf[:, 2, :, :][hps_conf_pos_bool[:, 2, :, :]].unsqueeze(-1)
        dt_conf = torch.cat([dt_conf_1, dt_conf_2, dt_conf_3], axis=1)
        loss_unvis_conf = F.binary_cross_entropy(dt_conf, dt_conf * 0)
        ##############################
    elif vis_obj_num != 0 and unvis_obj_num != 0:
        ##############################
        # kps coord loss
        gt_x1 = gt[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        gt_y1 = gt[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        gt_x2 = gt[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        gt_y2 = gt[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        gt_x3 = gt[:, 4, :, :][hps_coord_pos_bool[:, 4, :, :]].unsqueeze(-1)
        gt_y3 = gt[:, 5, :, :][hps_coord_pos_bool[:, 5, :, :]].unsqueeze(-1)

        gt_x1y1 = torch.cat([gt_x1, gt_y1], axis=1)
        gt_x2y2 = torch.cat([gt_x2, gt_y2], axis=1)
        gt_x3y3 = torch.cat([gt_x3, gt_y3], axis=1)

        gt_xy_1 = torch.cat([gt_x1y1, gt_x2y2, gt_x3y3], axis=1)
        gt_xy_2 = torch.cat([gt_x2y2, gt_x3y3, gt_x1y1], axis=1)
        gt_xy_3 = torch.cat([gt_x3y3, gt_x1y1, gt_x2y2], axis=1)

        mask_x1 = hps_vis_pos[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        mask_y1 = hps_vis_pos[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        mask_x2 = hps_vis_pos[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        mask_y2 = hps_vis_pos[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        mask_x3 = hps_vis_pos[:, 4, :, :][hps_coord_pos_bool[:, 4, :, :]].unsqueeze(-1)
        mask_y3 = hps_vis_pos[:, 5, :, :][hps_coord_pos_bool[:, 5, :, :]].unsqueeze(-1)

        mask_x1y1 = torch.cat([mask_x1, mask_y1], axis=1)
        mask_x2y2 = torch.cat([mask_x2, mask_y2], axis=1)
        mask_x3y3 = torch.cat([mask_x3, mask_y3], axis=1)

        mask_xy_1 = torch.cat([mask_x1y1, mask_x2y2, mask_x3y3], axis=1)
        mask_xy_2 = torch.cat([mask_x2y2, mask_x3y3, mask_x1y1], axis=1)
        mask_xy_3 = torch.cat([mask_x3y3, mask_x1y1, mask_x2y2], axis=1)

        dt_x1 = pred_coord[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_y1 = pred_coord[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_x2 = pred_coord[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        dt_y2 = pred_coord[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        dt_x3 = pred_coord[:, 4, :, :][hps_coord_pos_bool[:, 4, :, :]].unsqueeze(-1)
        dt_y3 = pred_coord[:, 5, :, :][hps_coord_pos_bool[:, 5, :, :]].unsqueeze(-1)
        dt_xy = torch.cat([dt_x1, dt_y1, dt_x2, dt_y2, dt_x3, dt_y3], axis=1)
        dt_xy_1 = dt_xy * mask_xy_1
        dt_xy_2 = dt_xy * mask_xy_2
        dt_xy_3 = dt_xy * mask_xy_3

        loss1 = torch.abs(dt_xy_1 - gt_xy_1)
        loss2 = torch.abs(dt_xy_2 - gt_xy_2)
        loss3 = torch.abs(dt_xy_3 - gt_xy_3)

        gt_xy_1 = gt_xy_1.unsqueeze(1)
        gt_xy_2 = gt_xy_2.unsqueeze(1)
        gt_xy_3 = gt_xy_3.unsqueeze(1)
        gt_xy_cat = torch.cat([gt_xy_1, gt_xy_2, gt_xy_3], axis=1)

        dt_xy_1 = dt_xy_1.unsqueeze(1)
        dt_xy_2 = dt_xy_2.unsqueeze(1)
        dt_xy_3 = dt_xy_3.unsqueeze(1)
        dt_xy_cat = torch.cat([dt_xy_1, dt_xy_2, dt_xy_3], axis=1)

        loss1_sum = torch.sum(loss1, dim=-1).unsqueeze(-1)
        loss2_sum = torch.sum(loss2, dim=-1).unsqueeze(-1)
        loss3_sum = torch.sum(loss3, dim=-1).unsqueeze(-1)
        loss_sum_cat = torch.cat([loss1_sum, loss2_sum, loss3_sum], axis=-1)

        first_pt_gt = torch.argmin(loss_sum_cat, axis=-1)
        first_pt_gt_onehot = F.one_hot(first_pt_gt, num_classes=3)

        pred_coord_pt_gt = dt_xy_cat[first_pt_gt_onehot == 1]
        target_coord_pt_gt = gt_xy_cat[first_pt_gt_onehot == 1]
        loss_coord = F.l1_loss(pred_coord_pt_gt, target_coord_pt_gt, size_average=False)
        loss_coord = loss_coord / (mask_xy_1.sum() + 1e-4)
        ##############################
        # kps vis conf loss
        gt_vis_1 = torch.cat([mask_x1, mask_x2, mask_x3], axis=1)
        gt_vis_2 = torch.cat([mask_x2, mask_x3, mask_x1], axis=1)
        gt_vis_3 = torch.cat([mask_x3, mask_x1, mask_x2], axis=1)

        dt_conf_1 = pred_conf[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_conf_2 = pred_conf[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_conf_3 = pred_conf[:, 2, :, :][hps_conf_pos_bool[:, 2, :, :]].unsqueeze(-1)
        dt_conf = torch.cat([dt_conf_1, dt_conf_2, dt_conf_3], axis=1)
        dt_vis_1 = F.sigmoid(dt_conf) * gt_vis_1
        dt_vis_2 = F.sigmoid(dt_conf) * gt_vis_2
        dt_vis_3 = F.sigmoid(dt_conf) * gt_vis_3

        gt_vis_1 = gt_vis_1.unsqueeze(1)
        gt_vis_2 = gt_vis_2.unsqueeze(1)
        gt_vis_3 = gt_vis_3.unsqueeze(1)
        gt_vis_ts_cat = torch.cat([gt_vis_1, gt_vis_2, gt_vis_3], axis=1)

        dt_vis_1 = dt_vis_1.unsqueeze(1)
        dt_vis_2 = dt_vis_2.unsqueeze(1)
        dt_vis_3 = dt_vis_3.unsqueeze(1)
        dt_vis_cat = torch.cat([dt_vis_1, dt_vis_2, dt_vis_3], axis=1)

        pred_vis_pt_gt = dt_vis_cat[first_pt_gt_onehot == 1]
        target_vis_pt_gt = gt_vis_ts_cat[first_pt_gt_onehot == 1]
        loss_vis_conf = F.binary_cross_entropy(pred_vis_pt_gt, target_vis_pt_gt)
        ##############################
        # kps unvis conf loss
        mask_unvis_x1 = hps_unvis_pos[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        mask_unvis_x2 = hps_unvis_pos[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)
        mask_unvis_x3 = hps_unvis_pos[:, 2, :, :][hps_conf_pos_bool[:, 2, :, :]].unsqueeze(-1)

        gt_unvis_1 = torch.cat([mask_unvis_x1, mask_unvis_x2, mask_unvis_x3], axis=1)
        gt_unvis_2 = torch.cat([mask_unvis_x2, mask_unvis_x3, mask_unvis_x1], axis=1)
        gt_unvis_3 = torch.cat([mask_unvis_x3, mask_unvis_x1, mask_unvis_x2], axis=1)

        dt_unvis_1 = F.sigmoid(dt_conf) * gt_unvis_1
        dt_unvis_2 = F.sigmoid(dt_conf) * gt_unvis_2
        dt_unvis_3 = F.sigmoid(dt_conf) * gt_unvis_3

        gt_unvis_1 = gt_unvis_1.unsqueeze(1)
        gt_unvis_2 = gt_unvis_2.unsqueeze(1)
        gt_unvis_3 = gt_unvis_3.unsqueeze(1)
        gt_unvis_cat = torch.cat([gt_unvis_1, gt_unvis_2, gt_unvis_3], axis=1)

        dt_unvis_1 = dt_unvis_1.unsqueeze(1)
        dt_unvis_2 = dt_unvis_2.unsqueeze(1)
        dt_unvis_3 = dt_unvis_3.unsqueeze(1)
        dt_unvis_cat = torch.cat([dt_unvis_1, dt_unvis_2, dt_unvis_3], axis=1)

        pred_unvis_pt_gt = dt_unvis_cat[first_pt_gt_onehot == 1]
        target_unvis_pt_gt = gt_unvis_cat[first_pt_gt_onehot == 1]
        loss_unvis_conf = F.binary_cross_entropy(pred_unvis_pt_gt, target_unvis_pt_gt * 0)
        ##############################
    return loss_coord, loss_vis_conf + loss_unvis_conf


def hps_cyc2_loss(pred_coord, pred_conf, gt, hps_pos, hps_vis_pos, hps_unvis_pos):
    hps_coord_pos_bool = hps_pos.gt(0)
    hps_conf_pos_bool = hps_pos[:, [0, 2], :, :].gt(0)
    hps_vis_pos_bool = hps_vis_pos.gt(0)
    hps_unvis_pos_bool = hps_unvis_pos.gt(0)
    vis_obj_num = hps_vis_pos_bool.sum()
    unvis_obj_num = hps_unvis_pos_bool.sum()
    if vis_obj_num == 0 and unvis_obj_num == 0:
        loss_coord = torch.tensor(0.0).to(pred_coord.device)
        loss_vis_conf = torch.tensor(0.0).to(pred_coord.device)
        loss_unvis_conf = torch.tensor(0.0).to(pred_coord.device)
    elif vis_obj_num != 0 and unvis_obj_num == 0:
        loss_unvis_conf = torch.tensor(0.0).to(pred_coord.device)
        ##############################
        # kps coord loss
        gt_x1 = gt[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        gt_y1 = gt[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        gt_x2 = gt[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        gt_y2 = gt[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)

        gt_x1y1 = torch.cat([gt_x1, gt_y1], axis=1)
        gt_x2y2 = torch.cat([gt_x2, gt_y2], axis=1)

        gt_xy_1 = torch.cat([gt_x1y1, gt_x2y2], axis=1)
        gt_xy_2 = torch.cat([gt_x2y2, gt_x1y1], axis=1)

        mask_x1 = hps_vis_pos[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        mask_y1 = hps_vis_pos[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        mask_x2 = hps_vis_pos[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        mask_y2 = hps_vis_pos[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)

        mask_x1y1 = torch.cat([mask_x1, mask_y1], axis=1)
        mask_x2y2 = torch.cat([mask_x2, mask_y2], axis=1)

        mask_xy_1 = torch.cat([mask_x1y1, mask_x2y2], axis=1)
        mask_xy_2 = torch.cat([mask_x2y2, mask_x1y1], axis=1)

        dt_x1 = pred_coord[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_y1 = pred_coord[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_x2 = pred_coord[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        dt_y2 = pred_coord[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        dt_xy = torch.cat([dt_x1, dt_y1, dt_x2, dt_y2], axis=1)
        dt_xy_1 = dt_xy * mask_xy_1
        dt_xy_2 = dt_xy * mask_xy_2

        loss1 = torch.abs(dt_xy_1 - gt_xy_1)
        loss2 = torch.abs(dt_xy_2 - gt_xy_2)

        gt_xy_1 = gt_xy_1.unsqueeze(1)
        gt_xy_2 = gt_xy_2.unsqueeze(1)
        gt_xy_cat = torch.cat([gt_xy_1, gt_xy_2], axis=1)

        dt_xy_1 = dt_xy_1.unsqueeze(1)
        dt_xy_2 = dt_xy_2.unsqueeze(1)
        dt_xy_cat = torch.cat([dt_xy_1, dt_xy_2], axis=1)

        loss1_sum = torch.sum(loss1, dim=-1).unsqueeze(-1)
        loss2_sum = torch.sum(loss2, dim=-1).unsqueeze(-1)
        loss_sum_cat = torch.cat([loss1_sum, loss2_sum], axis=-1)

        first_pt_gt = torch.argmin(loss_sum_cat, axis=-1)
        first_pt_gt_onehot = F.one_hot(first_pt_gt, num_classes=2)

        pred_coord_pt_gt = dt_xy_cat[first_pt_gt_onehot == 1]
        target_coord_pt_gt = gt_xy_cat[first_pt_gt_onehot == 1]
        loss_coord = F.l1_loss(pred_coord_pt_gt, target_coord_pt_gt, size_average=False)
        loss_coord = loss_coord / (mask_xy_1.sum() + 1e-4)
        ##############################
        # kps vis conf loss
        gt_vis_1 = torch.cat([mask_x1, mask_x2], axis=1)
        gt_vis_2 = torch.cat([mask_x2, mask_x1], axis=1)

        dt_conf_1 = pred_conf[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_conf_2 = pred_conf[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_conf = torch.cat([dt_conf_1, dt_conf_2], axis=1)
        dt_vis_1 = F.sigmoid(dt_conf) * gt_vis_1
        dt_vis_2 = F.sigmoid(dt_conf) * gt_vis_2

        gt_vis_1 = gt_vis_1.unsqueeze(1)
        gt_vis_2 = gt_vis_2.unsqueeze(1)
        gt_vis_ts_cat = torch.cat([gt_vis_1, gt_vis_2], axis=1)

        dt_vis_1 = dt_vis_1.unsqueeze(1)
        dt_vis_2 = dt_vis_2.unsqueeze(1)
        dt_vis_cat = torch.cat([dt_vis_1, dt_vis_2], axis=1)

        pred_vis_pt_gt = dt_vis_cat[first_pt_gt_onehot == 1]
        target_vis_pt_gt = gt_vis_ts_cat[first_pt_gt_onehot == 1]
        loss_vis_conf = F.binary_cross_entropy(pred_vis_pt_gt, target_vis_pt_gt)
        ##############################
    elif vis_obj_num == 0 and unvis_obj_num != 0:
        loss_coord = torch.tensor(0.0).to(pred_coord.device)
        loss_vis_conf = torch.tensor(0.0).to(pred_coord.device)
        ##############################
        # kps unvis conf loss
        dt_conf_1 = pred_conf[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_conf_2 = pred_conf[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_conf = torch.cat([dt_conf_1, dt_conf_2], axis=1)
        loss_unvis_conf = F.binary_cross_entropy(F.sigmoid(dt_conf), dt_conf * 0)
        ##############################
    elif vis_obj_num != 0 and unvis_obj_num != 0:
        ##############################
        # kps coord loss
        gt_x1 = gt[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        gt_y1 = gt[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        gt_x2 = gt[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        gt_y2 = gt[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)

        gt_x1y1 = torch.cat([gt_x1, gt_y1], axis=1)
        gt_x2y2 = torch.cat([gt_x2, gt_y2], axis=1)

        gt_xy_1 = torch.cat([gt_x1y1, gt_x2y2], axis=1)
        gt_xy_2 = torch.cat([gt_x2y2, gt_x1y1], axis=1)

        mask_x1 = hps_vis_pos[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        mask_y1 = hps_vis_pos[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        mask_x2 = hps_vis_pos[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        mask_y2 = hps_vis_pos[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)

        mask_x1y1 = torch.cat([mask_x1, mask_y1], axis=1)
        mask_x2y2 = torch.cat([mask_x2, mask_y2], axis=1)

        mask_xy_1 = torch.cat([mask_x1y1, mask_x2y2], axis=1)
        mask_xy_2 = torch.cat([mask_x2y2, mask_x1y1], axis=1)

        dt_x1 = pred_coord[:, 0, :, :][hps_coord_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_y1 = pred_coord[:, 1, :, :][hps_coord_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_x2 = pred_coord[:, 2, :, :][hps_coord_pos_bool[:, 2, :, :]].unsqueeze(-1)
        dt_y2 = pred_coord[:, 3, :, :][hps_coord_pos_bool[:, 3, :, :]].unsqueeze(-1)
        dt_xy = torch.cat([dt_x1, dt_y1, dt_x2, dt_y2], axis=1)
        dt_xy_1 = dt_xy * mask_xy_1
        dt_xy_2 = dt_xy * mask_xy_2

        loss1 = torch.abs(dt_xy_1 - gt_xy_1)
        loss2 = torch.abs(dt_xy_2 - gt_xy_2)

        gt_xy_1 = gt_xy_1.unsqueeze(1)
        gt_xy_2 = gt_xy_2.unsqueeze(1)
        gt_xy_cat = torch.cat([gt_xy_1, gt_xy_2], axis=1)

        dt_xy_1 = dt_xy_1.unsqueeze(1)
        dt_xy_2 = dt_xy_2.unsqueeze(1)
        dt_xy_cat = torch.cat([dt_xy_1, dt_xy_2], axis=1)

        loss1_sum = torch.sum(loss1, dim=-1).unsqueeze(-1)
        loss2_sum = torch.sum(loss2, dim=-1).unsqueeze(-1)
        loss_sum_cat = torch.cat([loss1_sum, loss2_sum], axis=-1)

        first_pt_gt = torch.argmin(loss_sum_cat, axis=-1)
        first_pt_gt_onehot = F.one_hot(first_pt_gt, num_classes=2)

        pred_coord_pt_gt = dt_xy_cat[first_pt_gt_onehot == 1]
        target_coord_pt_gt = gt_xy_cat[first_pt_gt_onehot == 1]
        loss_coord = F.l1_loss(pred_coord_pt_gt, target_coord_pt_gt, size_average=False)
        loss_coord = loss_coord / (mask_xy_1.sum() + 1e-4)
        ##############################
        # kps vis conf loss
        gt_vis_1 = torch.cat([mask_x1, mask_x2], axis=1)
        gt_vis_2 = torch.cat([mask_x2, mask_x1], axis=1)

        dt_conf_1 = pred_conf[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        dt_conf_2 = pred_conf[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)
        dt_conf = torch.cat([dt_conf_1, dt_conf_2], axis=1)
        dt_vis_1 = F.sigmoid(dt_conf) * gt_vis_1
        dt_vis_2 = F.sigmoid(dt_conf) * gt_vis_2

        gt_vis_1 = gt_vis_1.unsqueeze(1)
        gt_vis_2 = gt_vis_2.unsqueeze(1)
        gt_vis_ts_cat = torch.cat([gt_vis_1, gt_vis_2], axis=1)

        dt_vis_1 = dt_vis_1.unsqueeze(1)
        dt_vis_2 = dt_vis_2.unsqueeze(1)
        dt_vis_cat = torch.cat([dt_vis_1, dt_vis_2], axis=1)

        pred_vis_pt_gt = dt_vis_cat[first_pt_gt_onehot == 1]
        target_vis_pt_gt = gt_vis_ts_cat[first_pt_gt_onehot == 1]
        loss_vis_conf = F.binary_cross_entropy(pred_vis_pt_gt, target_vis_pt_gt)
        ##############################
        # kps unvis conf loss
        mask_unvis_x1 = hps_unvis_pos[:, 0, :, :][hps_conf_pos_bool[:, 0, :, :]].unsqueeze(-1)
        mask_unvis_x2 = hps_unvis_pos[:, 1, :, :][hps_conf_pos_bool[:, 1, :, :]].unsqueeze(-1)

        gt_unvis_1 = torch.cat([mask_unvis_x1, mask_unvis_x2], axis=1)
        gt_unvis_2 = torch.cat([mask_unvis_x2, mask_unvis_x1], axis=1)

        dt_unvis_1 = F.sigmoid(dt_conf) * gt_unvis_1
        dt_unvis_2 = F.sigmoid(dt_conf) * gt_unvis_2

        gt_unvis_1 = gt_unvis_1.unsqueeze(1)
        gt_unvis_2 = gt_unvis_2.unsqueeze(1)
        gt_unvis_cat = torch.cat([gt_unvis_1, gt_unvis_2], axis=1)

        dt_unvis_1 = dt_unvis_1.unsqueeze(1)
        dt_unvis_2 = dt_unvis_2.unsqueeze(1)
        dt_unvis_cat = torch.cat([dt_unvis_1, dt_unvis_2], axis=1)

        pred_unvis_pt_gt = dt_unvis_cat[first_pt_gt_onehot == 1]
        target_unvis_pt_gt = gt_unvis_cat[first_pt_gt_onehot == 1]
        loss_unvis_conf = F.binary_cross_entropy(pred_unvis_pt_gt, target_unvis_pt_gt * 0)
        ##############################
    return loss_coord, loss_vis_conf + loss_unvis_conf
