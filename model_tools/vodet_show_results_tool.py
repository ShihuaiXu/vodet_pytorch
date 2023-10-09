import cv2
import numpy as np

color_list = np.array(
    [
        [0, 0, 255],  # red
        [0, 255, 0],  # grren
        [255, 0, 0],  # blue
        [0, 255, 255],  # yellow
        [255, 255, 0],
        [0, 127, 127],
        [36, 36, 36]
    ], dtype=np.int
)


def ctdet_show_result(data_args, image, results, bbox_thresh, show_txt):
    multi_index = data_args['multi_index']
    multi_num = len(multi_index)
    class_num = results.shape[1]
    # i means class index
    for i in range(class_num):
        c = color_list[i+multi_num].tolist()
        # 0 represents batch size
        for result in results[0][i]:
            bbox, conf = result[:4], result[4]
            if conf >= bbox_thresh:
                bbox = np.array(bbox, dtype=np.int32)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 1)
                if show_txt:
                    txt = '{}'.format(data_args['class_names'][i+multi_num])
                    num_x, num_y = str(conf).split('.')
                    txt = txt + '_' + str(num_x) + '.' + str(num_y[0])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cat_size = cv2.getTextSize(txt, font, 0.5, 1)[0]

                    cv2.rectangle(image, (bbox[0], bbox[1] - cat_size[1] - 2), (bbox[0] + cat_size[0], bbox[1] - 2), c,
                                  -1)
                    cv2.putText(image, txt, (bbox[0], bbox[1] - 2), font, 0.5, (0, 0, 0), thickness=1,
                                lineType=cv2.LINE_AA)


def multi_show_result(data_args, image, results, bbox_thresh, pt_thresh, show_txt):
    for i, result_temp in enumerate(results):
        c = color_list[i].tolist()
        # 0 represents batch size
        for result in result_temp[0]:
            bbox, conf, hps_merge = result[:4], result[4], result[5:]
            if conf >= bbox_thresh:
                bbox = np.array(bbox, dtype=np.int32)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 1)
                if show_txt:
                    txt = '{}'.format(data_args['class_names'][i])
                    num_x, num_y = str(conf).split('.')
                    txt = txt + '_' + str(num_x) + '.' + str(num_y[0])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cat_size = cv2.getTextSize(txt, font, 0.5, 1)[0]

                    cv2.rectangle(image, (bbox[0], bbox[1] - cat_size[1] - 2), (bbox[0] + cat_size[0], bbox[1] - 2), c,
                                  -1)
                    cv2.putText(image, txt, (bbox[0], bbox[1] - 2), font, 0.5, (0, 0, 0), thickness=1,
                                lineType=cv2.LINE_AA)
                # only while bbox > thresh, pts will be output
                # pts conf, start from 7, bbox(4), score(1), hps_merge(x1, y1, conf1, x2, y2, conf2, ...)
                for j in range(0, hps_merge.size, 3):
                    if hps_merge[j + 2] >= pt_thresh:
                        cv2.circle(image, (int(hps_merge[j]), int(hps_merge[j + 1])), 2, c, -1)
                        cv2.putText(image, str(j // 3) + ': ' + str(hps_merge[j + 2])[:3],
                                    (int(hps_merge[j]), int(hps_merge[j + 1])), font, 0.5, c, thickness=1,
                                    lineType=cv2.LINE_AA)


def show_results(data_args, out_full_path, image, ctdet_results, multi_results, bbox_thresh=0.5, pt_thresh=0.3,
                 show_txt=True):
    ctdet_show_result(data_args, image, ctdet_results, bbox_thresh, show_txt)
    multi_show_result(data_args, image, multi_results, bbox_thresh, pt_thresh, show_txt)
    cv2.imwrite(out_full_path, image)
