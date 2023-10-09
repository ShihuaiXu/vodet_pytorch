import torch
from utils.utils import nms, topk_channel, transpose_and_gather_feat


def ctdet_decode(hm_feat, wh_feat, reg_feat, K=20):
    batch, cat, height, width = hm_feat.size()
    hm_feat_nms = nms(hm_feat)
    scores, inds, ys, xs = topk_channel(hm_feat_nms, K=K)
    reg = transpose_and_gather_feat(reg_feat, inds)
    xs = xs.view(batch, cat, K, 1) + reg[:, :, :, 0:1]
    ys = ys.view(batch, cat, K, 1) + reg[:, :, :, 1:2]
    wh = transpose_and_gather_feat(wh_feat, inds)

    scores = scores.view(batch, cat, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=3)
    detections = torch.cat([bboxes, scores], dim=3)
    return detections


def multi_decode(multi_pts_num, hm_feat, wh_feat, reg_feat, hps_coord_feat, hps_conf_feat, hm_hp_feat=None,
                 hm_hp_offset_feat=None, pt_thresh=0.3, bbox_prop=0.3, K=20):
    batch, cat, height, width = hm_feat.size()
    detections = []
    for i in range(cat):
        hm_feat_temp = hm_feat[:, i:i + 1, :, :]
        hm_feat_temp_nms = nms(hm_feat_temp)
        scores, inds, ys, xs = topk_channel(hm_feat_temp_nms, K=K)

        pt_num = multi_pts_num[i]
        hps_coord = transpose_and_gather_feat(hps_coord_feat[:, :pt_num * 2, :, :], inds)
        hps_coord[..., ::2] += xs.view(batch, 1, K, 1).expand(batch, 1, K, pt_num)
        hps_coord[..., 1::2] += ys.view(batch, 1, K, 1).expand(batch, 1, K, pt_num)
        hps_conf = transpose_and_gather_feat(hps_conf_feat[:, :pt_num, :, :], inds)

        hps_coord = hps_coord.view(batch, K, pt_num, 2).permute(0, 2, 1, 3).contiguous()
        hps_conf = hps_conf.view(batch, K, pt_num, 1).permute(0, 2, 1, 3).contiguous()
        hps_merge = torch.cat([hps_coord, hps_conf], dim=3)

        reg = transpose_and_gather_feat(reg_feat, inds)
        xs = xs.view(batch, 1, K, 1) + reg[:, :, :, 0:1]
        ys = ys.view(batch, 1, K, 1) + reg[:, :, :, 1:2]
        wh = transpose_and_gather_feat(wh_feat, inds)
        scores = scores.view(batch, 1, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=3)

        if hm_hp_feat is not None:
            hm_hp_feat_temp = hm_hp_feat[i]
            hm_hp_feat_temp_nms = nms(hm_hp_feat_temp)
            hm_score, hm_inds, hm_ys, hm_xs = topk_channel(hm_hp_feat_temp_nms, K=K)
            hm_hp_offset = transpose_and_gather_feat(hm_hp_offset_feat, hm_inds)
            hm_xs += hm_hp_offset[:, :, :, 0]
            hm_ys += hm_hp_offset[:, :, :, 1]

            mask = (hm_score > pt_thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_pts = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(batch, pt_num, K, K, 2)
            hps_pts = hps_coord.unsqueeze(3).expand(batch, pt_num, K, K, 2)

            dist = (((hps_pts - hm_pts) ** 2).sum(dim=4) ** 0.5)
            min_dist, min_ind = dist.min(dim=3)
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, pt_num, K, 1, 1).expand(batch, pt_num, K, 1, 2)
            hm_pts = hm_pts.gather(3, min_ind)
            hm_pts = hm_pts.view(batch, pt_num, K, 2)

            l = bboxes[:, :, :, 0:1].expand(batch, pt_num, K, 1)
            t = bboxes[:, :, :, 1:2].expand(batch, pt_num, K, 1)
            r = bboxes[:, :, :, 2:3].expand(batch, pt_num, K, 1)
            b = bboxes[:, :, :, 3:4].expand(batch, pt_num, K, 1)

            mask = (hm_pts[..., 0:1] < l) + (hm_pts[..., 0:1] > r) + \
                   (hm_pts[..., 1:2] < t) + (hm_pts[..., 1:2] > b) + \
                   (min_dist > (torch.max(b - t, r - l) * bbox_prop))
            mask = (mask > 0).float().expand(batch, pt_num, K, 3)
            hm_pts = torch.cat((hm_pts, hm_score), 3)
            hps_merge = (1 - mask) * hm_pts + mask * hps_merge
            hps_merge = hps_merge.permute(0, 2, 1, 3).contiguous().view(batch, 1, K, pt_num * 3)
        detections.append(torch.cat([bboxes, scores, hps_merge], dim=3).squeeze(1))

    return detections


def post_parse(output, data_args, args):
    with torch.no_grad():
        torch.cuda.synchronize()
        output['hm'] = output['hm'].sigmoid_()
        multi_pts_num = data_args['multi_pts_num']
        multi_num = len(multi_pts_num)
        ctdet_hm_feat = output['hm'][:, multi_num:, :, :]
        multi_hm_feat = output['hm'][:, 0:multi_num, :, :]
        wh_feat = output['wh']
        reg_feat = output['reg']
        hps_coord_feat = output['hps_coord']
        hps_conf_feat = output['hps_conf'].sigmoid_()
        hm_hp_car4_feat = output['hm_hp_car'].sigmoid_()
        hm_hp_pil3_feat = output['hm_hp_pil'].sigmoid_()
        hm_hp_ride2_feat = output['hm_hp_ride'].sigmoid_()
        hm_hp_feat = [hm_hp_car4_feat, hm_hp_pil3_feat, hm_hp_ride2_feat]
        hm_hp_offset_feat = output['hm_hp_offset']

        ctdet_dets = ctdet_decode(ctdet_hm_feat, wh_feat, reg_feat)
        multi_dets = multi_decode(multi_pts_num, multi_hm_feat, wh_feat, reg_feat, hps_coord_feat, hps_conf_feat,
                                  hm_hp_feat, hm_hp_offset_feat, pt_thresh=args.pts_thresh, bbox_prop=args.bbox_prop)
        return ctdet_dets, multi_dets
