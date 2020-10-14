import torch
import numpy as np
from scipy import linalg
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score
from model.loss import kld_gauss
from scipy.optimize import linear_sum_assignment

eps = np.finfo(float).eps


def sklearn_f1(output, target):
    output = output.argmax(dim=-1).cpu().data.numpy()
    target = target.cpu().data.numpy()
    return f1_score(target, output, average='micro')


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def f1(yt_hat, yt, n_class, return_rp=False):
    output = yt_hat.squeeze(-1)
    target = yt.squeeze(-1)
    real_pred = torch.zeros_like(output)
    for i in range(n_class):
        idx = output == i
        label = target[idx]
        if len(label) == 0:
            continue
        real_pred[idx] = torch.mode(label)[0]

     
    if return_rp:
        return real_pred
    else:
        # return real_pred.eq(target).sum().float().div(len(real_pred)).item()
        real_pred = real_pred.cpu().data.numpy()
        target = target.cpu().data.numpy()
        return f1_score(target, real_pred, average='micro')


def acc_family(yt_hat, yt, yf, tf_map):
    '''tf_map: dictionary that maps indices from instrument to family'''
    yt_hat = acc(yt_hat, yt, n_class=12, return_rp=True)
    yf_hat = torch.LongTensor([tf_map[i.item()] for i in yt_hat])
    yf_hat = yf_hat.cpu().data.numpy()
    yf = yf.cpu().data.numpy()
    # return yf_hat.eq(yf.squeeze(-1)).sum().float().div(len(yf_hat)).item()
    return f1_score(yf, yf_hat, average='micro')

def FID(output, target, model):
    model.eval()
    output = model.get_feat(output).cpu().data.numpy()
    target = model.get_feat(target).cpu().data.numpy()
    mu_output = np.atleast_1d(np.mean(output, axis=0))
    sigma_output = np.atleast_2d(np.cov(output, rowvar=False))
    mu_target = np.atleast_1d(np.mean(target, axis=0))
    sigma_target = np.atleast_2d(np.cov(target, rowvar=False))
    mu1, sigma1 = mu_output, sigma_output
    mu2, sigma2 = mu_target, sigma_target

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def nmi(output, target):
    output = output.squeeze(-1).cpu().data.numpy()
    target = target.squeeze(-1).cpu().data.numpy()
    return normalized_mutual_info_score(output, target, average_method='arithmetic')


def cluster_acc(Y_pred, Y):
    Y_pred, Y = Y_pred.squeeze(-1).cpu().data.numpy(), Y.squeeze(-1).cpu().data.numpy()
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max()-w)
    return sum([w[row[i],col[i]] for i in range(row.shape[0])]) * 1.0/Y_pred.size
