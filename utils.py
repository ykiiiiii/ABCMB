import torch
from torch import nn, optim
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import matplotlib.pyplot as plt
import torch.fft


def torch_ps(image, device):
    eps = 1e-6
    npix = image.shape[-1]
    image = image.view(-1, npix, npix)
    fourier_image = torch.fft.fftn(image, dim=(-2, -1))
    fourier_amplitudes = torch.abs(fourier_image) ** 2
    kfreq = torch.fft.fftfreq(npix) * npix
    kfreq2D = torch.meshgrid(kfreq, kfreq)
    knrm = torch.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = knrm.flatten()
    fourier_amplitudes = torch.flatten(fourier_amplitudes, start_dim=1)
    # kbins = torch.arange(0.5, npix // 2 + 1, 1.0)
    kbins = torch.arange(0.5, 31, 1.0)
    q = []
    for i in range(31):  # len(kbins)
        if i == 0:
            pass
        else:
            select = (knrm >= kbins[i - 1]) * (knrm < kbins[i])
            q.append(torch.mean(fourier_amplitudes[:, select], dim=1))
    ans = torch.stack(q).permute(1, 0)
    ans *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2).to(device)
    return ans  # [batch,bins_number]


def str2r(ID):
    start = ID.index("r = ")  
    end = ID.index("<start>")  
    # Store omega and sigma
    y = np.float(ID[start + 4 : end])
    return y

