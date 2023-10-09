def ctdet_coord_map(down_ratio, dets):
    dets = dets.detach().cpu().numpy()
    dets[:, :, :, :4] *= down_ratio
    return dets


def multi_coord_map(down_ratio, dets):
    for i in range(len(dets)):
        dets[i] = dets[i].detach().cpu().numpy()
        # bbox
        dets[i][:, :, :4] *= down_ratio
        # pts coords, start from 5, bbox(4), score(1), hps_merge(x1, y1, conf1, x2, y2, conf2, ...)
        for j in range(5, dets[i].shape[-1], 3):
            dets[i][:, :, j:j + 2] *= down_ratio
    return dets


def coord_map(data_args, img_size, ctdet_dets, multi_dets):
    down_ratio_net = data_args['downsample_ratio']
    down_ratio_img = img_size[0] / data_args['input_size'][0]
    down_ratio = down_ratio_net * down_ratio_img

    ctdet_results = ctdet_coord_map(down_ratio, ctdet_dets)
    multi_results = multi_coord_map(down_ratio, multi_dets)
    return ctdet_results, multi_results
