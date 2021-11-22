import torch
import numpy as np
import sys

def z_score(dataset):
    mean = torch.mean(dataset)
    std = torch.std(dataset)
    z_norm = (dataset - mean) / std
    return z_norm

def math_mase_loss(preds, labels):
    error = torch.abs(preds - labels)
    series_size_inv = 1 / (error.size(0) - 1)
    horizon_error = torch.mean(error, 0)
    timely_diff_label = torch.abs(labels[:-1, :, :] - labels[1:, :, :])
    # print(preds.size())
    # sys.exit()
    mase = torch.div(horizon_error, torch.mean(timely_diff_label, 0))
    # print(horizon_error[5, :])
    # print(torch.mean(timely_diff_label, 0)[5, :])
    mase = torch.mean(mase, 1)
    # print(mase)
    # print(torch.mean(timely_diff_label))
    # sys.exit()
    mase = mase.data.cpu().numpy()
    # print(mase)
    # print(horizon_error)
    # print(torch.sum(timely_diff_label, (0,2)))
    # sys.exit()
    return mase

def math_mad_loss(preds, labels):
    error = torch.abs(preds - labels)
    mad = [np.median(torch.abs(error)[:, tt, :].detach().cpu().numpy()) for tt in range(error.size(1))]
    return mad

