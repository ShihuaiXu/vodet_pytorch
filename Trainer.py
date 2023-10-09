import time
import torch
from progress.bar import Bar
from model_tools.vodet_model_with_loss_tools import Vodet_ModleWithLoss
from utils.data_parallel import DataParallel
from utils.utils import AverageMeter


class Trainer(object):
    def __init__(self, args, data_args, net_args, model, optimizer):
        self.args = args
        self.optimizer = optimizer
        self.loss_stats = ['loss', 'hm_loss', 'wh_loss', 'reg_loss', 'hm_det_loss', 'wh_det_loss', 'reg_det_loss',
                           'hps_coord_loss', 'hps_conf_loss', 'hm_hp_loss', 'hm_hp_off_loss']
        self.model_with_loss = Vodet_ModleWithLoss(model, data_args, net_args)

    # set the conv weight and tensor related to loss to the device(cpu or cuda)
    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def run_epoch(self, phase, epoch, data_loader):
        # get the obj for network forward and calculate loss
        model_with_loss = self.model_with_loss
        ##############################
        # set the train/val status
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.args.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
        ##############################
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar(max=num_iters)
        end = time.time()

        for iter_id, batch in enumerate(data_loader):
            if phase == 'train':
                # set for warm up lr
                if epoch < self.args.wp_epoch:
                    tmp_lr = self.args.lr * pow((iter_id + epoch * num_iters) * 1. / (self.args.wp_epoch * num_iters), 4)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = tmp_lr
                elif epoch == self.args.wp_epoch and iter_id == 0:
                    tmp_lr = self.args.lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = tmp_lr
                ##############################
            data_time.update(time.time() - end)
            # put the dataset to cuda
            for i in batch:
                if isinstance(batch[i], list):
                    for j in range(len(batch[i])):
                        batch[i][j] = batch[i][j].to(device=self.args.device, non_blocking=True)
                else:
                    batch[i] = batch[i].to(device=self.args.device, non_blocking=True)
            ##############################
            # update the info in screen
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} |lr: {lr:.4e}'. \
                format(epoch, iter_id, num_iters, phase=phase, total=bar.elapsed_td, eta=bar.eta_td,
                       lr=(self.optimizer.param_groups[0]['lr']))
            ##############################
            # zero the gradient
            if phase == 'train':
                self.optimizer.zero_grad()
            ##############################
            # model forward and calculate loss
            loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            ##############################
            # update the current loss in the screen
            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                if l != 'loss':
                    l_simpify = l[: -5]
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l_simpify, avg_loss_stats[l].avg)
                else:
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            ##############################
            # backpropagation computes the gradient
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            ##############################
            # one batch finished
            bar.next()
            ##############################
        # one epoch finished
        bar.finish()
        ##############################
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        return ret
