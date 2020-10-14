import argparse
from pathlib import Path
import json
import os
import math
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from parse_config import ConfigParser
from utils import set_seed, read_json
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from sklearn.metrics import f1_score
from data.audio_processor import *


spec_ext = ExtractSpectrogram(sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL, mode='mel')

class DataTransform:
    def get_idx(self, at_time=0.2):
        compensate_duration = 0.05
        load_duration = at_time + compensate_duration
        idx_comp = int(compensate_duration * 1.0 * SR / HOP)
        n_sample = int(load_duration * SR)
        desired_idx = int((1.0 * n_sample) / HOP) - idx_comp
        return n_sample, desired_idx

    def __call__(self, x):
        n_sample, desired_idx = self.get_idx()
        x = LoadNpArray(n_sample=n_sample)(x)
        x = ToTensor()(x)
        x = spec_ext(x)
        x = LogCompress()(x)
        x = Clipping(clip_min=X_MIN, clip_max=X_MAX)(x)
        x = MinMaxNorm(x_min=X_MIN, x_max=X_MAX)(x)
        x = x[:, :, desired_idx]
        return x


def get_data(x):
    for i, x_i in enumerate(x):
        x = data_transform(x_i)
        if i == 0:
            x_ori_cat = x
        else:
            x_ori_cat = torch.cat([x_ori_cat, x])
    return x_ori_cat


set_seed()

def eval_metrics(config):
    save_path = str(Path(config.resume).parent)
    print("The results will be saved under %s." % save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trsfm = transforms.Compose([DataTransform()])
    config['data_loader']['args']['batch_size'] = 64
    data_loader = config.init_obj('data_loader', module_data)
    data_loader.dataset.transform = trsfm
    valid_data_loader = data_loader.split_validation()

    model = config.init_obj('arch', module_arch)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    txt_file_name = 'met.txt' 
    txt_dir = os.path.join(save_path, txt_file_name)
    txt_file = open(txt_dir, 'w')

    for batch_idx, (x, idx, y) in enumerate(valid_data_loader):
        with torch.no_grad():
            y = torch.stack(y, dim=1).to(device)
            yt, yp, yc, yd, yf = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, -1:]
            x, h, mu, logvar, z_t, z_p, logits, prob = model(x.squeeze(1).to(device), yp, determine=True)
            yp_hat = torch.argmax(prob, dim=-1, keepdim=True)

        if batch_idx == 0:
            yp_cat = yp
            yp_hat_cat = yp_hat
        else:
            yp_cat = torch.cat([yp_cat, yp])
            yp_hat_cat = torch.cat([yp_hat_cat, yp_hat])

    f1 = module_metric.f1(yp_hat_cat, yp_cat, n_class=82)
    acc = module_metric.cluster_acc(yp_hat_cat, yp_cat)
    nmi = module_metric.nmi(yp_hat_cat, yp_cat)
        
    return f1, acc, nmi 


def batch_eval_metrics(model_folder, w_config, save_path):
    assert len(w_config.split('-')) == 6
    print("Weight config: %s" % w_config)
    metrics = ['f1', 'acc', 'nmi']
    # metrics = ['f1', 'acc']
    mets = {k: [] for k in metrics}
    avg_mets = {k: 0 for k in metrics}
    model_count = 0
    eval_models = [f for f in Path(model_folder).glob('*') if w_config in str(f)]

    txt_file_name = 'batch_eval_met_%s.txt' % w_config
    txt_dir = os.path.join(save_path, txt_file_name)
    txt_file = open(txt_dir, 'w')
  
    for f in eval_models:
        # for d in f.glob('*'):
        #     m = d / 'model_best.pth'
        #     if m.is_file():
        #         f = d; txt_file.write('%s\n' % f)
        #         break
        txt_file.write('%s\n' % f)
        print("Processing model: %s" % f)
        config_file = read_json(str(f / 'config.json'))
        model_best = str(f / 'model_best.pth')
        config = ConfigParser(config_file, resume=model_best, testing=True)
        f1, acc, nmi = eval_metrics(config)
        for m, s in zip(metrics, [f1, acc, nmi]):
            mets[m].append(s)
            avg_mets[m] += s
        model_count += 1
    for k in avg_mets.keys():
        avg_mets[k] /= model_count

    txt_file.write(json.dumps(mets, indent=2))
    txt_file.write('\n')
    txt_file.write(json.dumps(avg_mets, indent=2))
    txt_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_folder', type=str)
    parser.add_argument('-w', '--w_config', type=str)
    parser.add_argument('-s', '--save_path', type=str)
    args = parser.parse_args()

    batch_eval_metrics(args.model_folder, args.w_config, args.save_path)
