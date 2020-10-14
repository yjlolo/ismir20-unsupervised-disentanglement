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

def get_pyc(model, zt, pitch_clfr):
    n_sample = len(zt)
    n_pitch = 82
    output = np.zeros([n_pitch, n_pitch])
    for i in range(n_pitch):
        zp = model.emb.weight[i]
        x = model.decode(zt, zp.repeat(zt.shape[0], 1))[0]
        yi_hat = F.softmax(pitch_clfr(x), dim=-1).mean(dim=0)  # prediction of clfr
        output[i] = yi_hat.data.cpu().numpy()
    return output
        

def eval_sample(config, config_clfr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = str(Path(config.resume).parent)
    print("The results will be saved under %s." % save_path)
    trsfm = transforms.Compose([DataTransform()])
    data_loader = config.init_obj('data_loader', module_data)
    data_loader.dataset.transform = trsfm
    valid_data_loader = data_loader.split_validation()

    model = config.init_obj('arch', module_arch)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    clfr = config_clfr.init_obj('arch', module_arch)
    clfr_checkpoint = torch.load(config_clfr.resume)
    clfr_state_dict = clfr_checkpoint['state_dict']
    clfr.load_state_dict(clfr_state_dict)
    clfr = clfr.to(device)
    clfr.eval()

    txt_file_name = 'rand_sample.txt' 
    txt_dir = os.path.join(save_path, txt_file_name)
    txt_file = open(txt_dir, 'w')

    for batch_idx, (x, idx, y) in enumerate(valid_data_loader):
        with torch.no_grad():
            x = x.squeeze(1).to(device)
            y = torch.stack(y, dim=1).to(device)
            yt, yp, yc, yd, yf = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, -1:]
            x_hat, h, mu, logvar, z_t, z_p, logits, prob = model(x, yp, determine=True)
            yp_hat = torch.argmax(prob, dim=-1, keepdim=True)

        if batch_idx == 0:
            yp_cat = yp
            yp_hat_cat = yp_hat
            zt_cat = z_t
            zp_cat = z_p
            x_hat_cat = x_hat
            x_cat = x
        else:
            yp_cat = torch.cat([yp_cat, yp])
            yp_hat_cat = torch.cat([yp_hat_cat, yp_hat])
            zt_cat = torch.cat([zt_cat, z_t])
            zp_cat = torch.cat([zp_cat, z_p])
            x_hat_cat = torch.cat([x_hat_cat, x_hat])
            x_cat = torch.cat([x_cat, x])

    zt_rand = torch.distributions.normal.Normal(0, 1).sample(zt_cat.size()).to(device)
    x_hat_rand, _ = model.decode(zt_rand, zp_cat)

    fid_recon = module_metric.FID(x_hat_cat, x_cat, clfr)  # upper-bound, FID for reconstruction
    fid_rand = module_metric.FID(x_hat_rand, x_cat, clfr)
    # get the mapping from prediction to true pitch
    # pyc = get_pyc(model, zt_cat, clfr)
    # cd_score = module_metric.con_div(pyc)

    return fid_recon, fid_rand 


def batch_eval_sample(model_folder, w_config, save_path, clfr_folder):
    assert len(w_config.split('-')) == 6
    print("Weight config: %s" % w_config)
    metrics = ['fid_recon', 'fid_rand']
    mets = {k: [] for k in metrics}
    avg_mets = {k: 0 for k in metrics}
    model_count = 0
    eval_models = [f for f in Path(model_folder).glob('*') if w_config in str(f)]

    txt_file_name = 'batch_eval_sample_%s.txt' % w_config
    txt_dir = os.path.join(save_path, txt_file_name)
    txt_file = open(txt_dir, 'w')

    clfr_config = read_json(str(Path(clfr_folder) / 'config.json'))
    clfr = str(Path(clfr_folder) / 'model_best.pth')
    clfr_config = ConfigParser(clfr_config, resume=clfr, testing=True)
  
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
        fid_recon, fid_rand = eval_sample(config, clfr_config)
        for m, s in zip(metrics, [fid_recon, fid_rand]):
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
    parser.add_argument('-c', '--classifier', type=str)
    args = parser.parse_args()

    batch_eval_sample(args.model_folder, args.w_config, args.save_path, args.classifier)
