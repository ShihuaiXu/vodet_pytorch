import argparse
import os
import yaml
import torch
import cv2
import pycocotools.coco as coco

from models import get_model
from utils.utils import load_model, get_images
from model_tools.vodet_preprocess_tool import pre_process
from model_tools.vodet_postparse_tool import post_parse
from model_tools.vodet_coord_map_tool import coord_map
from model_tools.vodet_save_text_tool import save_text

parser = argparse.ArgumentParser(description='vodet pytorch train parser')
parser.add_argument('--net_config', default='./configs/net_configs/res18_dlaup_apa_head.yaml',
                    help='network setting config')
parser.add_argument('--data_config', default='./configs/data_configs/vodet_data_config.yaml',
                    help='data setting config')
parser.add_argument('--load_model', default='', help='path to model')
parser.add_argument('--out_txt_path', default='output_txt', help='output txt path')
parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
parser.add_argument('--json_test', default='', help='path to json file')
parser.add_argument('--image_path', default='', help='image path')
parser.add_argument('--bbox_thresh', type=float, default=0.5, help='bbox threshold')
parser.add_argument('--pts_thresh', type=float, default=0.3, help='pts threshold')
parser.add_argument('--bbox_prop', type=float, default=0.3, help='bbox_prop')
args = parser.parse_args()

if len(args.gpus) > 1:
    assert 0, 'just support one gpu inference!!!'

for config_path in [args.net_config, args.data_config]:
    if not os.path.exists(config_path):
        assert 0, 'config file is not found !!!'

if not os.path.exists(args.load_model):
    assert 0, 'model path is not exists!!!'

if not os.path.exists(args.json_test):
    assert 0, 'json_test path is not exists!!!'

if args.image_path != '' and not os.path.exists(args.image_path):
    assert 0, 'img path is not exists!!!'

if not os.path.isdir(args.out_txt_path):
    os.makedirs(args.out_txt_path)


def model_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')
    # read data & net config
    stream = open(args.data_config, mode='r', encoding='utf-8')
    data_args = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    stream = open(args.net_config, mode='r', encoding='utf-8')
    net_args = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    ##############################
    # initialize the model
    model_path = args.load_model
    model = eval('get_model.' + net_args['model'])(net_args)
    model = load_model(model, model_path)
    model = model.to(args.device)
    model.eval()
    ##############################
    image_names = get_images(args.json_test, args.image_path)

    for index, image_name in enumerate(image_names):
        print(str(index + 1) + '/' + str(len(image_names)))
        out_full_path = os.path.join(args.out_txt_path, image_name.rsplit('/', 1)[-1].split('.')[0] + '.txt')
        # data preprocess
        image_draw = cv2.imread(image_name)
        height, width = image_draw.shape[:2]
        image_infer = pre_process(image_draw, data_args)
        image_infer = image_infer.to(args.device)
        torch.cuda.synchronize()
        ##############################
        # model infer
        output = model(image_infer)
        # post parse
        ctdet_dets, multi_dets = post_parse(output, data_args, args)
        # feat size coordinate map to origin image size
        ctdet_results, multi_results = coord_map(data_args, [width, height], ctdet_dets, multi_dets)
        # save detection results
        save_text(data_args, out_full_path, ctdet_results, multi_results, args.bbox_thresh, args.pts_thresh)


if __name__ == '__main__':
    model_test()
