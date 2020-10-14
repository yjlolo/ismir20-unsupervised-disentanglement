import random
import os
import argparse
import collections
import torch
from torchvision import transforms
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils import set_seed
from parse_config import ConfigParser
from trainer.trainer_clfr import ClassifierTrainer 
from experiment.eval_metrics import *


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

set_seed()

def main(config):
    logger = config.get_logger('train')
    trsfm = transforms.Compose([DataTransform()])
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    data_loader.dataset.transform = trsfm
    # data_loader.val_idx = np.load('/data/yinjyun/datasets/sol/acidsInstruments-ordinario/data/valid_idx.npy')
    valid_data_loader = data_loader.split_validation()
    dd = np.setdiff1d(valid_data_loader.sampler.indices,
                      np.load('/data/yinjyun/datasets/sol/acidsInstruments-ordinario/data/valid_idx.npy'))
    assert len(dd) == 0

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    model.apply(weights_init)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # criterion = config.init_ftn('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    trainer = ClassifierTrainer(model, criterion, metrics, optimizer,
                             config=config,
                             data_loader=data_loader,
                             valid_data_loader=valid_data_loader,
                             lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
