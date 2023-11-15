import os
import cv2
import math
import numpy as np
import random
import pycocotools.coco as coco
from torch.utils.data import Dataset

from dataset.utils import gaussian_radius, draw_umich_gaussian as draw_gaussian
from dataset.data_aug.color_augmentation import ColorAugmentation
from dataset.data_aug.affine_augmentation import AffineAugmentation


class Vodet_Dataset(Dataset):
    # initialize parameters for dataset
    def __init__(self, split, data_args):
        super(Vodet_Dataset, self).__init__()
        self.split = split
        self.input_w, self.input_h = data_args['input_size']
        self.downsample_ratio = data_args['downsample_ratio']
        self.output_w, self.output_h = self.input_w // self.downsample_ratio, self.input_h // self.downsample_ratio
        self.mean, self.std = np.array(data_args['preprocess']['mean'], dtype=np.float32), np.array(
            data_args['preprocess']['std'], dtype=np.float32)
        self.directs = {'front': 1, 'rear': 0, 'left': 3, 'right': 2}
        self.class_names = data_args['class_names']
        self.num_classes = len(self.class_names)
        self.multi_index = data_args['multi_index']
        self.multi_names = [self.class_names[x] for x in self.multi_index]
        self.multi_pts_num = data_args['multi_pts_num']
        self.max_pts_num = max(data_args['multi_pts_num'])
        self.pts_center_index = data_args['pts_center_index']
        self.cat_ids = {i + 1: i for i, v in enumerate(data_args['class_names'])}
        if split == 'train':
            self.annot_path = data_args['json_train_file']
        elif split == 'val':
            self.annot_path = data_args['json_val_file']
        self.img_dir = data_args['image_path']
        self.coco = coco.COCO(self.annot_path)
        self.image_ids = self.coco.getImgIds()
        self.image_ids_directs = self.split_image_directs()
        self.affine_p = data_args['data_aug']['affine_p']
        self.flip_p = data_args['data_aug']['flip_p']
        self.flip_idx = data_args['data_aug']['flip_idx']
        self.overlay_p = data_args['data_aug']['overlay_p']
        self.fr_resize_r = data_args['data_aug']['fr_resize_r']
        self.lr_resize_r = data_args['data_aug']['lr_resize_r']
        self.fr_crop_s = data_args['data_aug']['fr_crop_s']
        self.lr_crop_s = data_args['data_aug']['lr_crop_s']
        self.crop_min_s = data_args['data_aug']['crop_min_s']
        self.AffineAugmentation = AffineAugmentation(data_args['data_aug']['min_scale'],
                                                     data_args['data_aug']['max_scale'],
                                                     data_args['data_aug']['rot'],
                                                     [self.input_w, self.input_h],
                                                     [self.output_w, self.output_h],
                                                     data_args['data_filter']['area_size'],
                                                     data_args['data_filter']['area_ratio'])
        print('Loaded {} {} samples'.format(split, len(self.image_ids)))

    # must be override
    def __getitem__(self, index):
        img_data, bboxes, pts, labels = self.pull_item(index)
        # data preprocess
        inp_data = img_data.astype(np.float32)
        inp_data = (inp_data - self.mean) / self.std
        inp_data = inp_data.transpose(2, 0, 1)
        ##############################
        ret = self.label_encode(bboxes, pts, labels)
        ret.update({'input': inp_data})
        return ret

    # must be override
    def __len__(self):
        return len(self.image_ids)

    # img, label, dataaug, filter
    def pull_item(self, index):
        img_id = self.image_ids[index]
        img_data, img_shape, file_name, anns = self.get_image_info(img_id)
        # bboxes in (xmin，ymin), (xmax，ymax) format
        bboxes, pts, labels = self.get_ann_info(anns, img_shape)
        c = np.array([img_shape[0] / 2., img_shape[1] / 2.], dtype=np.float32)
        s = max(img_shape[0], img_shape[1]) * 1.0

        if self.overlay_p:
            if np.random.random() <= self.overlay_p:
                img_data, file_name, bboxes, pts, labels = self.overlay(file_name, img_data, img_shape, bboxes, pts, labels)
            else:
                img_data = self.img_resize_crop(img_data, img_shape, self.fr_resize_r, self.fr_crop_s)
                bboxes, pts, labels = self.label_resize_crop(bboxes, pts, labels, self.fr_resize_r, self.fr_crop_s)
        if np.random.random() < self.flip_p:
            img_data, bboxes, pts = self.flip(img_data, bboxes, pts, labels, c)
        if np.random.random() < self.affine_p:
            img_data, bboxes, pts, labels = self.AffineAugmentation(img_data, bboxes, pts, labels, c, s)
        # self.visualize_label(img_data, file_name, bboxes, pts, labels)
        return img_data, bboxes, pts, labels

    # put the label to the feature map
    def label_encode(self, bboxes, pts, labels):
        output_h, output_w = self.output_h, self.output_w
        max_pts_num = self.max_pts_num
        # initialize the ndarray for gts
        hm = np.zeros((len(self.multi_index), output_h, output_w), dtype=np.float32)
        hm_ind = np.zeros((1, output_h, output_w), dtype=np.float32)
        wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((2, output_h, output_w), dtype=np.float32)

        hm_det = np.zeros((self.num_classes - len(self.multi_index), output_h, output_w), dtype=np.float32)
        hm_det_ind = np.zeros((1, output_h, output_w), dtype=np.float32)
        wh_det = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg_det = np.zeros((2, output_h, output_w), dtype=np.float32)

        hps_coord = [np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                     np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                     np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                     np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                     np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32)]

        hps_ind = [np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                   np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                   np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                   np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32),
                   np.zeros((self.max_pts_num * 2, output_h, output_w), dtype=np.float32)]

        hps_vis_ind = [np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                       np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                       np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                       np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                       np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32)]

        hps_unvis_ind = [np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                         np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                         np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                         np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32),
                         np.zeros((self.max_pts_num, output_h, output_w), dtype=np.float32)]

        for name, pt_num in zip(self.multi_names, self.multi_pts_num):
            vars()['hm_hp_' + name] = np.zeros((pt_num, output_h, output_w), dtype=np.float32)
        hm_hp_ind = [np.zeros((2, output_h, output_w), dtype=np.float32),
                     np.zeros((2, output_h, output_w), dtype=np.float32),
                     np.zeros((2, output_h, output_w), dtype=np.float32),
                     np.zeros((2, output_h, output_w), dtype=np.float32),
                     np.zeros((2, output_h, output_w), dtype=np.float32)]

        hm_hp_offset = [np.zeros((2, output_h, output_w), dtype=np.float32),
                        np.zeros((2, output_h, output_w), dtype=np.float32),
                        np.zeros((2, output_h, output_w), dtype=np.float32),
                        np.zeros((2, output_h, output_w), dtype=np.float32),
                        np.zeros((2, output_h, output_w), dtype=np.float32)]
        ##############################
        for i, (bbox, pt, label) in enumerate(zip(bboxes, pts, labels)):
            if label in self.multi_index:
                if label in self.pts_center_index:
                    pt_sum_x, pt_sum_y, count = 0, 0, 0
                    for j in range(len(pt)):
                        if pt[j][2] == 2:
                            pt_sum_x += pt[j][0]
                            pt_sum_y += pt[j][1]
                            count += 1
                    if count >= 2:
                        ct = np.array([pt_sum_x / count, pt_sum_y / count], dtype=np.float32)
                        ct_int = ct.astype(np.int32)
                        center_x, center_y = ct_int
                        hm_ind[:, center_y, center_x] = 1
                        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                        reg[:, center_y, center_x] = ct - ct_int
                        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                        radius = max(0, int(radius))
                        draw_gaussian(hm[label], ct_int, radius)

                        num_pt = pt.shape[0]
                        pt_int = pt.astype(np.int32)
                        for k in range(0, max_pts_num):
                            if k < num_pt and pt[k][2] != 0:
                                hps_coord[label][k * 2, center_y, center_x] = pt[k][0] - center_x
                                hps_coord[label][k * 2 + 1, center_y, center_x] = pt[k][1] - center_y
                                hps_ind[label][k * 2: k * 2 + 2, center_y, center_x] = 1
                                hps_vis_ind[label][k, center_y, center_x] = 1
                                draw_gaussian(vars()['hm_hp_' + self.class_names[label]][k], pt_int[k, :2], radius)
                                hm_hp_offset[label][:, pt_int[k, 1], pt_int[k, 0]] = pt[k, :2] - pt_int[k, :2]
                                hm_hp_ind[label][:, pt_int[k, 1], pt_int[k, 0]] = 1
                            else:
                                hps_unvis_ind[label][k, center_y, center_x] = 1
                else:
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    center_x, center_y = ct_int
                    hm_ind[:, center_y, center_x] = 1
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    wh[:, center_y, center_x] = w, h
                    reg[:, center_y, center_x] = ct - ct_int
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    draw_gaussian(hm[label], ct_int, radius)

                    num_pt = pt.shape[0]
                    pt_int = pt.astype(np.int32)
                    for j in range(0, max_pts_num):
                        if j < num_pt and pt[j][2] != 0:
                            hps_coord[label][j * 2, center_y, center_x] = pt[j][0] - center_x
                            hps_coord[label][j * 2 + 1, center_y, center_x] = pt[j][1] - center_y
                            hps_ind[label][j * 2: j * 2 + 2, center_y, center_x] = 1
                            hps_vis_ind[label][j, center_y, center_x] = 1
                            draw_gaussian(vars()['hm_hp_' + self.class_names[label]][j], pt_int[j, :2], radius)
                            hm_hp_offset[label][:, pt_int[j, 1], pt_int[j, 0]] = pt[j, :2] - pt_int[j, :2]
                            hm_hp_ind[label][:, pt_int[j, 1], pt_int[j, 0]] = 1
                        else:
                            hps_unvis_ind[label][j, center_y, center_x] = 1
            else:
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                center_x, center_y = ct_int
                hm_det_ind[:, center_y, center_x] = 1
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                wh_det[:, center_y, center_x] = w, h
                reg_det[:, center_y, center_x] = ct - ct_int
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                draw_gaussian(hm_det[label - len(self.multi_index)], ct_int, radius)

        ret = {'hm': hm, 'hm_ind': hm_ind, 'wh': wh, 'reg': reg, 'hm_det': hm_det, 'hm_det_ind': hm_det_ind,
               'hps_ind': hps_ind, 'hps_vis_ind': hps_vis_ind, 'hps_unvis_ind': hps_unvis_ind, 'wh_det': wh_det,
               'reg_det': reg_det, 'hps_coord': hps_coord, 'hps_ind': hps_ind, 'hm_hp_offset': hm_hp_offset,
               'hm_hp_ind': hm_hp_ind}
        for name, pt_num in zip(self.multi_names, self.multi_pts_num):
            ret.update({'hm_hp_' + name: vars()['hm_hp_' + name]})
        return ret

    # visualization bbox and bbox after dataaug and filter
    def visualize_label(self, img, file_name, bboxes, pts, labels):
        for bbox, label in zip(bboxes, labels):
            cv2.rectangle(img, (int(bbox[0] * 4), int(bbox[1] * 4)), (int(bbox[2]) * 4, int(bbox[3]) * 4), (255, 0, 0),
                          1)
            cv2.putText(img, str(label), (int(bbox[0] * 4), int(bbox[1] * 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        for pt, label in zip(pts, labels):
            for i in range(pt.shape[0]):
                if pt[i][2] == 2:
                    cv2.circle(img, (int(pt[i][0] * 4), int(pt[i][1]) * 4), 3, (0, 0, 255), 1)
                    cv2.putText(img, str(i), (int(pt[i][0] * 4), int(pt[i][1] * 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(file_name, img)

    def flip(self, img_data, bboxes, pts, labels, c):
        img_flip_data = img_data[:, ::-1, :]
        width = img_data.shape[1]
        c[0] = width - c[0] - 1
        for bbox in bboxes:
            # label from 0 to 1279 while flip
            bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        for i, pt in enumerate(pts):
            for j in range(len(pt)):
                if pt[j][2] == 2:
                    pt[j][0] = width - pt[j][0] - 1
            if labels[i] in self.multi_index:
                for e in self.flip_idx[labels[i]]:
                    pt[e[0]], pt[e[1]] = pt[e[1]].copy(), pt[e[0]].copy()
        return img_flip_data, bboxes, pts

    # change the bbox from left, top, width, height to left, top, right, bottom
    def coco_bbox_decode(self, coco_bbox):
        bbox = np.array([coco_bbox[0], coco_bbox[1], coco_bbox[0] + coco_bbox[2], coco_bbox[1] + coco_bbox[3]],
                        dtype=np.float32)
        return bbox

    # avoid the bbox labeling out of image
    def clip_bbox(self, bbox, width, height):
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        return bbox

    # avoid the pt labeling out of image
    def clip_pt(self, pt, width, height):
        pt[:, 0] = np.clip(pt[:, 0], 0, width - 1)
        pt[:, 1] = np.clip(pt[:, 1], 0, height - 1)
        return pt

    # (front and rear) and (left and right) overlay
    def overlay(self, file_name, img_data, img_shape, bboxes, pts, labels):
        direct = file_name.split('_')[-1].split('.')[0]
        direct_index = self.directs[direct]

        img_id_pair = random.choice(self.image_ids_directs[direct_index])
        img_data_pair, img_shape_pair, file_name_pair, anns_pair = self.get_image_info(img_id_pair)
        bboxes_pair, pts_pair, labels_pair = self.get_ann_info(anns_pair, img_shape_pair)
        ##############################
        # images resize and crop and concat
        img_data = self.img_resize_crop(img_data, img_shape, self.lr_resize_r, self.lr_crop_s)
        img_data_pair = self.img_resize_crop(img_data_pair, img_shape_pair, self.lr_resize_r, self.lr_crop_s)
        img_data_concat = cv2.vconcat([img_data, img_data_pair])
        ##############################
        # labels resize and crop and concat
        half_height = img_shape[1] / 2
        bboxes, pts, labels = self.label_resize_crop(bboxes, pts, labels, self.lr_resize_r, self.lr_crop_s,
                                                     half_height)
        bboxes_pair, pts_pair, labels_pair = self.label_resize_crop(bboxes_pair, pts_pair, labels_pair,
                                                                    self.lr_resize_r, self.lr_crop_s, half_height,
                                                                    is_down=True)

        file_name_concat = file_name.split('.png')[0] + '_' + file_name_pair
        bboxes_concat = bboxes + bboxes_pair
        pts_concat = pts + pts_pair
        labels_concat = labels + labels_pair
        ##############################
        return img_data_concat, file_name_concat, bboxes_concat, pts_concat, labels_concat

    def split_image_directs(self):
        image_ids_directs = [[], [], [], []]
        image_ids = self.coco.getImgIds()
        for image_id in image_ids:
            file_name = self.coco.loadImgs(ids=[image_id])[0]['file_name']
            for i, direct in enumerate(self.directs):
                if direct in file_name:
                    image_ids_directs[i].append(image_id)
                    break
        return image_ids_directs

    def get_image_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_file = os.path.join(self.img_dir, file_name[1:] if file_name[0] == '/' else file_name)
        img_data = cv2.imread(img_file)
        file_name = file_name.rsplit('/', 1)[-1]
        return img_data, [img_data.shape[1], img_data.shape[0]], file_name, anns

    def get_ann_info(self, anns, img_shape):
        # put the anns to the different list
        bboxes, pts, labels = [], [], []
        for k in range(len(anns)):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            bboxes.append(self.clip_bbox(self.coco_bbox_decode(ann['bbox']), img_shape[0], img_shape[1]))
            if cls_id in self.multi_index:
                pts.append(self.clip_pt(np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3), img_shape[0],
                                        img_shape[1]))
            else:
                pts.append(np.array([]))
            labels.append(cls_id)
        return bboxes, pts, labels

    def img_resize_crop(self, img_data, img_shape, resize_r, crop_s):
        img_data = cv2.resize(img_data, (img_shape[0], int(img_shape[1] * resize_r)))
        img_data = img_data[crop_s:, :, :]
        return img_data

    def label_resize_crop(self, bboxes, pts, labels, resize_r, crop_s, half_height=0, is_down=False):
        bboxes_new, pts_new, labels_new = [], [], []
        ##############################
        # resize the bbox and pt y coordinate
        for bbox in bboxes:
            # xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox[1] *= resize_r
            bbox[3] *= resize_r
        for pt in pts:
            for pt_ in pt:
                # pt_x, pt_y, visible = pt_[0], pt_[1], pt_[2]
                if pt_[2] == 2:
                    pt_[1] *= resize_r
        ##############################
        # whether keep ann by ymin
        for i in range(len(bboxes)):
            keep_signal = 0
            # xmin, ymin, xmax, ymax = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
            ##############################
            # if (ymin < crop_s), (ymin, ymax and pt_y) - crop and save this obj
            if bboxes[i][1] > crop_s:
                keep_signal = 1
                bboxes[i][1] -= crop_s
                bboxes[i][3] -= crop_s
                if is_down:
                    bboxes[i][1] += half_height
                    bboxes[i][3] += half_height
                # whether the obj has keypoints
                if len(pts[i]) != 0:
                    for pt in pts[i]:
                        if pt[2] == 2:
                            pt[1] -= crop_s
                            if is_down:
                                pt[1] += half_height
            ##############################
            # if ymin < crop_s < ymax, represents this obj has been cut off
            elif bboxes[i][1] < crop_s < bboxes[i][3]:
                bboxes[i][3] = bboxes[i][3] - crop_s
                bboxes[i][1] = 0
                if is_down:
                    bboxes[i][1] += half_height
                    bboxes[i][3] += half_height
                # if the cut obj height > crop_min_s, save this obj
                if bboxes[i][3] > self.crop_min_s:
                    keep_signal = 1
                    # whether the obj has keypoints
                    if len(pts[i]) != 0:
                        for pt in pts[i]:
                            if pt[2] == 2:
                                if pt[1] > crop_s:
                                    pt[1] -= crop_s
                                    if is_down:
                                        pt[1] += half_height
                                else:
                                    pt[0], pt[1], pt[2] = [0, 0, 0]
            if keep_signal:
                bboxes_new.append(bboxes[i])
                pts_new.append(pts[i])
                labels_new.append(labels[i])
        ##############################
        return bboxes_new, pts_new, labels_new
