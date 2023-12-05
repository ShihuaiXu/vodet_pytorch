import torch

from losses.vodet_loss import hm_focal_loss, reg_l1_loss, hps_l1_loss, hps_conf_ce_loss, hps_cyc3_loss, hps_cyc2_loss
from utils.utils import sigmoid


class Vodet_ModleWithLoss(torch.nn.Module):
    def __init__(self, model, data_args, net_args, phase='train'):
        super(Vodet_ModleWithLoss, self).__init__()
        self.class_names = data_args['class_names']
        self.multi_index = data_args['multi_index']
        self.multi_heat_index = data_args['multi_heat_index']
        self.multi_heat_names = [self.class_names[x] for x in self.multi_heat_index]
        self.hm_focal_loss_weight = net_args['loss_weight']['hm_focal_loss_weight']
        self.wh_loss_weight = net_args['loss_weight']['wh_loss_weight']
        self.reg_loss_weight = net_args['loss_weight']['reg_loss_weight']
        self.hps_coord_loss_weight = net_args['loss_weight']['hps_coord_loss_weight']
        self.hps_conf_loss_weight = net_args['loss_weight']['hps_conf_loss_weight']
        self.hm_hp_loss_weight = net_args['loss_weight']['hm_hp_loss_weight']
        self.hm_hp_offset_loss_weight = net_args['loss_weight']['hm_hp_offset_loss_weight']
        self.model = model
        self.hm_focal_loss = hm_focal_loss
        self.reg_l1_loss = reg_l1_loss
        self.hps_l1_loss = hps_l1_loss
        self.hps_conf_ce_loss = hps_conf_ce_loss
        self.hps_cyc3_loss = hps_cyc3_loss
        self.hps_cyc2_loss = hps_cyc2_loss
        self.phase = phase

    def forward(self, batch):
        # model forward
        output = self.model(batch['input'])
        hm_loss, wh_loss, reg_loss = 0, 0, 0
        hm_det_loss, wh_det_loss, reg_det_loss = 0, 0, 0
        hps_coord_loss, hps_conf_loss, hm_hp_loss, hm_hp_off_loss = 0, 0, 0, 0

        output['hm'] = sigmoid(output['hm'])
        hm_loss += self.hm_focal_loss(output['hm'][:, :len(self.multi_index), :, :], batch['hm'])
        hm_det_loss += self.hm_focal_loss(output['hm'][:, len(self.multi_index):, :, :], batch['hm_det']) * 1.5

        for name in self.multi_heat_names:
            output['hm_hp_' + name] = sigmoid(output['hm_hp_' + name])
            hm_hp_loss += self.hm_focal_loss(output['hm_hp_' + name], batch['hm_hp_' + name])

        wh_loss += self.reg_l1_loss(output['wh'], batch['wh'], batch['hm_notpil_ind'])
        wh_det_loss += self.reg_l1_loss(output['wh'], batch['wh_det'], batch['hm_det_ind']) * 1.5
        reg_loss += self.reg_l1_loss(output['reg'], batch['reg'], batch['hm_ind'])
        reg_det_loss += self.reg_l1_loss(output['reg'], batch['reg_det'], batch['hm_det_ind']) * 1.5

        hps_coord_loss += self.hps_l1_loss(output['hps_coord'], batch['hps_coord'][0], batch['hps_vis_ind'][0])
        hps_coord_loss += self.hps_l1_loss(output['hps_coord'], batch['hps_coord'][4], batch['hps_vis_ind'][4])

        hps_conf_loss += self.hps_conf_ce_loss(output['hps_conf'], batch['hps_vis_ind'][0], batch['hps_unvis_ind'][0])
        hps_conf_loss += self.hps_conf_ce_loss(output['hps_conf'], batch['hps_vis_ind'][4], batch['hps_unvis_ind'][4])

        pil3_hps_coord_loss, pil3_hps_conf_loss = self.hps_cyc3_loss(output['hps_coord'][:, :6, :, :],
                                                                     output['hps_conf'][:, :3, :, :],
                                                                     batch['hps_coord'][1],
                                                                     batch['hps_ind'][1][:, :6, :, :],
                                                                     batch['hps_vis_ind'][1][:, :6, :, :],
                                                                     batch['hps_unvis_ind'][1][:, :6, :, :])
        pil2_hps_coord_loss, pil2_hps_conf_loss = self.hps_cyc2_loss(output['hps_coord'][:, :4, :, :],
                                                                     output['hps_conf'][:, :2, :, :],
                                                                     batch['hps_coord'][2],
                                                                     batch['hps_ind'][2][:, :4, :, :],
                                                                     batch['hps_vis_ind'][2][:, :4, :, :],
                                                                     batch['hps_unvis_ind'][1][:, :4, :, :])
        pily_hps_coord_loss, pily_hps_conf_loss = self.hps_cyc2_loss(output['hps_coord'][:, :4, :, :],
                                                                     output['hps_conf'][:, :2, :, :],
                                                                     batch['hps_coord'][3],
                                                                     batch['hps_ind'][3][:, :4, :, :],
                                                                     batch['hps_vis_ind'][3][:, :4, :, :],
                                                                     batch['hps_unvis_ind'][3][:, :4, :, :])
        hps_coord_loss += pil3_hps_coord_loss
        hps_coord_loss += pil2_hps_coord_loss
        hps_coord_loss += pily_hps_coord_loss
        hps_conf_loss += pil3_hps_conf_loss
        hps_conf_loss += pil2_hps_conf_loss
        hps_conf_loss += pily_hps_conf_loss

        hm_hp_off_loss += self.hps_l1_loss(output['hm_hp_offset'], batch['hm_hp_offset'][0], batch['hm_hp_ind'][0])

        loss = hm_loss * self.hm_focal_loss_weight + wh_loss * self.wh_loss_weight + reg_loss * self.reg_loss_weight + \
               hm_det_loss * self.hm_focal_loss_weight + wh_det_loss * self.wh_loss_weight + \
               reg_det_loss * self.reg_loss_weight + hps_coord_loss * self.hps_coord_loss_weight + \
               hps_conf_loss * self.hps_conf_loss_weight + hm_hp_loss * self.hm_hp_loss_weight + \
               hm_hp_off_loss * self.hm_hp_loss_weight
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'reg_loss': reg_loss,
                      'hm_det_loss': hm_det_loss, 'wh_det_loss': wh_det_loss, 'reg_det_loss': reg_det_loss,
                      'hps_coord_loss': hps_coord_loss, 'hps_conf_loss': hps_conf_loss, 'hm_hp_loss': hm_hp_loss,
                      'hm_hp_off_loss': hm_hp_off_loss}
        return loss, loss_stats
