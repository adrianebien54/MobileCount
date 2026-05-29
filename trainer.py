import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import time
import pdb


class Trainer():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer = optim.AdamW(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.i_tb = 0
        self.epoch = 0

        # Loss history for plotting
        self.train_loss_history = []   # (epoch, avg_loss) per epoch
        self.val_loss_history = []     # (epoch, val_loss) per validation

        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            print(f'Resuming from {cfg.RESUME_PATH}')
            latest_state = torch.load(cfg.RESUME_PATH, weights_only=False)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch        = latest_state['epoch'] + 1
            self.i_tb         = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path     = latest_state['exp_path']
            self.exp_name     = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp',
                                           resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch

            self.timer['train time'].tic()
            epoch_train_loss = self.train()
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()
            self.timer['train time'].toc(average=False)

            self.train_loss_history.append((epoch + 1, epoch_train_loss))
            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50', 'Tenebrio']:
                    self.validate_V1()
                elif self.data_mode == 'WE':
                    self.validate_V2()
                elif self.data_mode == 'GCC':
                    self.validate_V3()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self):
        self.net.train()
        losses = AverageMeter()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.net(img, gt_map)
            loss = self.net.loss
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item())

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' %
                      (self.epoch + 1, i + 1, loss.item(),
                       self.optimizer.param_groups[0]['lr'] * 10000,
                       self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA,
                    pred_map[0].sum().data / self.cfg_data.LOG_PARA))

        return losses.avg

    def validate_V1(self):
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        time_sample = 0
        step = 0

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.net.forward(img, gt_map)

                step += 1
                time_start1 = time.time()
                test_map = self.net.test_forward(img)
                time_end1 = time.time()
                time_sample += time_end1 - time_start1

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                pred_cnt = np.sum(pred_map) / self.cfg_data.LOG_PARA
                gt_count = np.sum(gt_map) / self.cfg_data.LOG_PARA

                losses.update(self.net.loss.item())
                maes.update(abs(gt_count - pred_cnt))
                mses.update((gt_count - pred_cnt) ** 2)
                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer,
                                self.restore_transform, img, pred_map, gt_map)

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.val_loss_history.append((self.epoch + 1, loss))
        self._save_loss_plot()

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch,
                                         self.i_tb, self.exp_path, self.exp_name,
                                         [mae, mse, loss], self.train_record, self.log_txt)
        print_summary(self.exp_name, [mae, mse, loss], self.train_record)
        print('\nForward Time: %fms' % (time_sample * 1000 / step))

    def validate_V2(self):
        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        roi_mask = []
        from datasets.WE.setting import cfg_data
        from scipy import io as sio
        for val_folder in cfg_data.VAL_FOLDER:
            roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH, 'test',
                                                       val_folder + '_roi.mat'))['BW'])

        for i_sub, i_loader in enumerate(self.val_loader, 0):
            mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data
                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    pred_map = self.net.forward(img, gt_map)
                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()
                    for i_img in range(pred_map.shape[0]):
                        pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA
                        losses.update(self.net.loss.item(), i_sub)
                        maes.update(abs(gt_count - pred_cnt), i_sub)
                    if vi == 0:
                        vis_results(self.exp_name, self.epoch, self.writer,
                                    self.restore_transform, img, pred_map, gt_map)

        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)

        self.val_loss_history.append((self.epoch + 1, loss))
        self._save_loss_plot()

        self.train_record = update_model(self.net, self.epoch, self.exp_path, self.exp_name,
                                         [mae, 0, loss], self.train_record, self.log_txt)
        print_WE_summary(self.log_txt, self.epoch, [mae, 0, loss], self.train_record, maes)

    def validate_V3(self):
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8),
                  'weather': AverageCategoryMeter(7)}
        c_mses = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8),
                  'weather': AverageCategoryMeter(7)}

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data
            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                pred_map = self.net.forward(img, gt_map)
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()
                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA
                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = (gt_count - pred_cnt) ** 2
                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.val_loss_history.append((self.epoch + 1, loss))
        self._save_loss_plot()

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch,
                                         self.i_tb, self.exp_path, self.exp_name,
                                         [mae, mse, loss], self.train_record, self.log_txt)
        print_GCC_summary(self.log_txt, self.epoch, [mae, mse, loss],
                          self.train_record, c_maes, c_mses)

    def _save_loss_plot(self):
        if not self.train_loss_history or not self.val_loss_history:
            return

        train_epochs, train_losses = zip(*self.train_loss_history)
        val_epochs, val_losses = zip(*self.val_loss_history)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_epochs, train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
        ax.plot(val_epochs, val_losses, label='Val Loss', color='darkorange',
                linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('MobileCount — Tenebrio Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(self.exp_path, self.exp_name, 'loss_curve.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
