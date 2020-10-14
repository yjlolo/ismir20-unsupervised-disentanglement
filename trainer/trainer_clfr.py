import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tsnecuda import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from test import *



class ClassifierTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))


        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        torch.manual_seed(1111)
        for batch_idx, (x, idx, gt) in enumerate(self.data_loader):
            x = x.squeeze(1)
            gt = torch.stack(gt, dim=1).to(self.device)

            self.optimizer.zero_grad()
            if self.model.target == 'instrument':
                y = gt[:, 0]
            elif self.model.target == 'pitch':
                y = gt[:, 1]

            output = self.model(x)
            loss = self.criterion(output, y)
            
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, y))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        self.writer.set_step(epoch, 'train')
        self.writer.add_scalar('loss', self.train_metrics.avg('loss'))
        for met in self.metric_ftns:
            self.writer.add_scalar(met.__name__, self.train_metrics.avg(met.__name__))


        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        torch.manual_seed(1111)
        with torch.no_grad():
            for batch_idx, (x, idx, gt) in enumerate(self.valid_data_loader):
                x = x.squeeze(1)
                gt = torch.stack(gt, dim=1).to(self.device)

                if self.model.target == 'instrument':
                    y = gt[:, 0]
                elif self.model.target == 'pitch':
                    y = gt[:, 1]
                output = self.model(x)
                
                loss = self.criterion(output, y)
                
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, y))

        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('loss', self.valid_metrics.avg('loss'))
        for met in self.metric_ftns:
            self.writer.add_scalar(met.__name__, self.valid_metrics.avg(met.__name__))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
