import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from dataset.vodet_dataset import Vodet_Dataset
from models import get_model
from Trainer import Trainer
from utils.utils import save_model
from utils.logger import Logger

parser = argparse.ArgumentParser(description='vodet pytorch train parser')
parser.add_argument('--net_config', default='./configs/net_configs/res18_dlaup_apa_head.yaml',
                    help='network setting config')
parser.add_argument('--data_config', default='./configs/data_configs/vodet_data_config.yaml',
                    help='data setting config')
parser.add_argument('--json_train_file', required=True, help='json file of train')
parser.add_argument('--json_val_file', required=True, help='json file of val')
parser.add_argument('--image_path', help='image path of train and val')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--num_epochs', type=int, default=140, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=2.5e-4, help='training learning rate')
parser.add_argument('--lr_step', type=str, default='90,120', help='drop learning rate by 10.')
parser.add_argument('--seed', type=int, default=7, help='random seed (default: 7)')
parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers for data loader')
parser.add_argument('--load_model', default='', help='path to resume model')
parser.add_argument('--resume', action='store_true',
                    help='resume an experiment, Reloaded the optimizer parameter and set load_model to model_last.pth')
parser.add_argument('--wp_epoch', type=int, default=2,
                    help='warmup epochs for large dataset, default is 1, for small dataset default is 5-10')
parser.add_argument('--val_intervals', type=int, default=10, help='number of epochs to run validation.')
parser.add_argument('--save_dir', default='/home/xushihuai/vodet_weights', help='output log path')
parser.add_argument('--save_all', action='store_true', help='save model to disk every val_intervals epochs.')
parser.add_argument('--metric', default='loss', help='main metric to save best model')
parser.add_argument('--output_path', default='output_imgs', help='output imgs path')
args = parser.parse_args()

for config_path in [args.net_config, args.data_config]:
    if not os.path.exists(config_path):
        assert 0, 'net config or data config is not found !!!'

args.lr_step = [int(i) for i in args.lr_step.split(',')]


def train():
    # initial seed and cuda setting
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')
    ##############################
    # read data & net config
    stream = open(args.data_config, mode='r', encoding='utf-8')
    data_args = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    data_args['json_train_file'] = args.json_train_file
    data_args['json_val_file'] = args.json_val_file
    data_args['image_path'] = args.image_path
    stream = open(args.net_config, mode='r', encoding='utf-8')
    net_args = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    ##############################
    # initialize the log obj
    logger = Logger(args, data_args, net_args)
    # pytorch dataset and dataload
    train_dataset = Vodet_Dataset('train', data_args)
    val_dataset = Vodet_Dataset('val', data_args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ##############################
    # get the network
    model = eval('get_model.' + net_args['model'])(net_args)
    ##############################
    # initial the optimizer
    if net_args['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=1e-4, momentum=0.9)
    elif net_args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    ##############################
    # follow the centernet setting, related to multi GPU training
    master_batch_size = args.batch_size // len(args.gpus)
    rest_batch_size = (args.batch_size - master_batch_size)
    chunk_sizes = [master_batch_size]
    for i in range(len(args.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(args.gpus) - 1)
        if i < rest_batch_size % (len(args.gpus) - 1):
            slave_chunk_size += 1
        chunk_sizes.append(slave_chunk_size)
    ##############################
    # initial the train, which includes network, loss, optimizer and so on
    trainer = Trainer(args, data_args, net_args, model, optimizer)
    trainer.set_device(args.gpus, chunk_sizes, args.device)
    ##############################
    # epoch loop
    print('Starting training...')
    best = 1e10
    start_epoch = 0
    num_epochs = args.num_epochs
    for epoch in range(start_epoch + 1, num_epochs + 1):
        mark = epoch if args.save_all else 'last'
        log_dict_train = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        logger.write('epoch: {:.4e} |'.format(optimizer.param_groups[0]['lr']))
        print('epoch: {} |'.format(epoch))
        print('lr: {:.4e} |'.format(optimizer.param_groups[0]['lr']))
        for k, v in log_dict_train.items():
            print('{} {:8f} | '.format(k, v))
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if args.val_intervals > 0 and epoch % args.val_intervals == 0:
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                print('{} {:8f} | '.format(k, v))
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
                save_model(os.path.join(logger.writer.log_dir, './model_{}.pth'.format(mark)), epoch, model, optimizer)
            if log_dict_val[args.metric] < best:
                best = log_dict_val[args.metric]
                save_model(os.path.join(logger.writer.log_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(os.path.join(logger.writer.log_dir, 'model_last.pth'), epoch, model, optimizer)
        logger.write('\n')
        if epoch in args.lr_step:
            save_model(os.path.join(logger.writer.log_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = args.lr * (0.1 ** (args.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()
    ##############################


if __name__ == '__main__':
    train()
