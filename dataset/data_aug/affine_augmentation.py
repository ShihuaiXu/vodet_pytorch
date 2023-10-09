import cv2
import numpy as np


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


class AffineAugmentation(object):
    def __init__(self, min_scale, max_scale, rot, input_size, output_size, area_size, area_ratio):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rot = rot
        self.input_w, self.input_h = input_size
        self.output_w, self.output_h = output_size
        self.area_size = area_size
        self.area_ratio = area_ratio

    def __call__(self, img_data, bboxes, pts, labels, c, s):
        # random set s & c
        s = s * np.random.choice(np.arange(self.min_scale, self.max_scale, 0.1))
        w_border = self._get_border(128, img_data.shape[1])
        h_border = self._get_border(128, img_data.shape[0])
        c[0] = np.random.randint(low=w_border, high=img_data.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img_data.shape[0] - h_border)
        ##############################
        # size for input image into network
        trans_input = get_affine_transform(c, s, self.rot, [self.input_w, self.input_h])
        # img affine transformation
        img_affi_data = cv2.warpAffine(img_data, trans_input, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        # size for coordinate in feature map encode
        trans_output = get_affine_transform(c, s, self.rot, [self.output_w, self.output_h])

        new_bboxes, new_pts, new_labels = [], [], []
        for bbox, pt, label in zip(bboxes, pts, labels):
            # bbox apply affine transformation
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            # visible pt apply affine transformation
            for i in range(0, pt.shape[0]):
                if pt[i][2] == 2:
                    pt[i][:2] = affine_transform(pt[i][:2], trans_output)
            new_data = self._clip_data(bbox, pt, label)
            if new_data is not None:
                new_bboxes.append(new_data[0])
                new_pts.append(new_data[1])
                new_labels.append(new_data[2])
        return img_affi_data, new_bboxes, new_pts, new_labels

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    # clip bbox and pts out of the featmap
    def _clip_data(self, bbox, pt, label):
        output_w, output_h = self.output_w, self.output_h
        h_before_clip, w_before_clip = bbox[3] - bbox[1], bbox[2] - bbox[0]
        bbox_area_before_clip = w_before_clip * h_before_clip
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        bbox_area_after_clip = w * h
        bbox_area_clip_ratio = bbox_area_after_clip / (bbox_area_before_clip + 1e-5)
        if h <= 0 or w <= 0 or bbox_area_after_clip < self.area_size[label] or \
                bbox_area_clip_ratio < self.area_ratio[label]:
            return None
        # set the out of feature map pt to be unvisible
        for i in range(pt.shape[0]):
            if pt[i][2] == 2:
                if pt[i][0] < 0 or pt[i][0] > output_w - 1 or pt[i][1] < 0 or pt[i][1] > output_h - 1:
                    pt[i][2] = 0
        return bbox, pt, label
