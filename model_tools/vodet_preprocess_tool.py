import numpy as np
import cv2
import torch


def pre_process(image, data_args):
    inp_width, inp_height = data_args['input_size']
    inp_image = cv2.resize(image, (inp_width, inp_height))
    mean, std = np.array(data_args['preprocess']['mean'], dtype=np.float32), \
                np.array(data_args['preprocess']['std'], dtype=np.float32)
    inp = inp_image.astype(np.float32)
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    inp = torch.from_numpy(inp)
    return inp
