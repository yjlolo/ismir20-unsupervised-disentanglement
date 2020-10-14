import argparse
from pathlib import Path
import os
import math
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from parse_config import ConfigParser
from utils import set_seed
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from sklearn.metrics import f1_score
from trainer.trainer import GMVAETrainer
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

def latent_classifier(config, target):
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

    if target == 'pitch':
        n_class = 82 
    elif target == 'family':
        n_class = 4
    elif target == 'instrument':
        n_class = 12
    elif target =='dynamic':
        n_class = 5

    latent_dim = config['arch']['args']['latent_dim']
    clf = nn.Sequential(
        nn.Linear(latent_dim, n_class),
        # nn.Linear(latent_dim, latent_dim * 4),
#         nn.ReLU(),
#         nn.Linear(latent_dim * 4, latent_dim * 4),
#         nn.ReLU(),
#         nn.Linear(latent_dim * 4, n_class)
    )
    clf = clf.to(device)

    txt_file_name = 'lc-t_%s.txt' % target 
    txt_dir = os.path.join(save_path, txt_file_name)
    txt_file = open(txt_dir, 'w')

    optimizer = optim.Adam(lr=0.003, params=clf.parameters())

    n_epoch = 1000
    # n_epoch = 1
    early_stop = 30
    best_loss = math.inf
    best_score = -math.inf
    n_not_improve = 0

    for epoch in range(1, n_epoch + 1):
        train_score = 0
        train_loss = 0
        for batch_idx, (x, idx, y) in enumerate(data_loader):
            with torch.no_grad():
                y = torch.stack(y, dim=1).to(device)
                yt, yp, yc, yd, yf = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
                if target == 'pitch':
                    y = yp
                elif target =='family':
                    y = yf
                elif target == 'instrument':
                    y = yt
                elif target == 'dynamic':
                    y = yd
                x, h, mu, logvar, z_t, z_p, logits, prob = model(x.squeeze(1).to(device), yp, determine=True)

            y_hat = clf(z_t)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx == 0:
                y_cat = y.cpu().data.numpy()
                y_hat_cat = y_hat.cpu().argmax(1).data.numpy()
            else:
                y_cat = np.hstack([y_cat, y.cpu().data.numpy()])
                y_hat_cat = np.hstack([y_hat_cat, y_hat.cpu().argmax(1).data.numpy()])           

        train_score += f1_score(y_cat, y_hat_cat, average='micro')
        train_loss /= len(data_loader)

        with torch.no_grad():
             valid_score = 0  
             valid_loss = 0
             for batch_idx, (x, idx, y) in enumerate(valid_data_loader):
                 y = torch.stack(y, dim=1).to(device)
                 yt, yp, yc, yd, yf = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
                 if target == 'pitch':
                     y = yp
                 elif target =='family':
                     y = yf
                 elif target == 'instrument':
                     y = yt
                 elif target == 'dynamic':
                     y = yd
                 
                 x, h, mu, logvar, z_t, z_p, logits, prob = model(x.squeeze(1).to(device), yp, determine=True)
                 y_hat = clf(z_t)
                 loss = F.cross_entropy(y_hat, y)
                 valid_loss += loss.item()
                 if batch_idx == 0:
                     y_cat = y.cpu().data.numpy()
                     y_hat_cat = y_hat.argmax(1).cpu().data.numpy()
                 else:
                     y_cat = np.hstack([y_cat, y.data.cpu().numpy()])
                     y_hat_cat = np.hstack([y_hat_cat, y_hat.argmax(1).cpu().data.numpy()])           

        valid_score += f1_score(y_cat, y_hat_cat, average='micro')
        valid_loss /= len(valid_data_loader)

        if valid_score >  best_score:
            torch.save(clf, os.path.join(save_path, 'lc-target_%s-f1.pth' % target))
            best_loss = valid_loss
            best_score = valid_score
            n_not_improve = 0
        else:
            n_not_improve += 1

        print('')
        print("Epoch %d" % epoch)
        print("Train score: %f" % train_score, "Train loss: %f" % train_loss)
        print("Valid score: %f" % valid_score, "Valid loss: %f" % valid_loss)
        print("Best score: %f" % best_score, "Best loss: %f" % best_loss)

        if n_not_improve >= early_stop:
            print("Performance has not improved for %d epochs, stop training." % early_stop)
            print("The best score: %f" % best_score)
            txt_file.write("Epoch %d \n" % epoch)
            txt_file.write("Train score: %f, Train loss: %f \n" % (train_score, train_loss))
            txt_file.write("Valid score: %f, Valid loss: %f \n" % (valid_score, valid_loss))
            txt_file.write("Best score: %f, Best loss: %f" % (best_score, best_loss))
            txt_file.close()
            break

    return best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str)
    parser.add_argument('-r', '--resume', default=None, type=str)
    parser.add_argument('-t', '--target', default=None, type=str)
    args = parser.parse_args()

    assert args.target in ['instrument', 'pitch', 'family', 'dynamic'] 

    config = ConfigParser.from_args(args, testing=True)
    latent_classifer(config, args.target)
