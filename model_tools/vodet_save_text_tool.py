import os
import numpy as np


def ctdet_save_text(data_args, out_file, results, bbox_thresh):
    multi_index = data_args['multi_index']
    multi_num = len(multi_index)
    class_num = results.shape[1]
    # i means class index
    for i in range(class_num):
        # 0 represents batch size
        for result in results[0][i]:
            bbox, conf = result[:4], result[4]
            if conf >= bbox_thresh:
                bbox = np.array(bbox, dtype=np.int32)
                line = str(i + multi_num) + ' ' + str(conf) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(
                    bbox[2]) + ' ' + str(bbox[3])
                out_file.write(line + '\n')


def multi_save_text(out_file, results, bbox_thresh, pt_thresh):
    for i, result_temp in enumerate(results):
        # 0 represents batch size
        for result in result_temp[0]:
            bbox, conf, hps_merge = result[:4], result[4], result[5:]
            if conf >= bbox_thresh:
                bbox = np.array(bbox, dtype=np.int32)
                line = str(i) + ' ' + str(conf) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(
                    bbox[2]) + ' ' + str(bbox[3])
                # only while bbox > thresh, pts will be output
                # pts conf, start from 7, bbox(4), score(1), hps_merge(x1, y1, conf1, x2, y2, conf2, ...)
                for j in range(0, hps_merge.size, 3):
                    if hps_merge[j + 2] >= pt_thresh:
                        line += ' ' + str(hps_merge[j]) + ' ' + str(hps_merge[j + 1]) + ' ' + '2'
                    else:
                        line += ' 0 0 0'
                out_file.write(line + '\n')


def save_text(data_args, out_full_path, ctdet_results, multi_results, bbox_thresh=0.5, pt_thresh=0.3):
    out_file = open(out_full_path, 'w')
    ctdet_save_text(data_args, out_file, ctdet_results, bbox_thresh=0.4)
    multi_save_text(out_file, multi_results, bbox_thresh=0.5, pt_thresh=0.3)
    out_file.close()
