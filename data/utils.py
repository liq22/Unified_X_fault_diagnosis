import torch
import numpy as np
def wgn2(x, snr):
    "加随机噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.randn(x.size()).cuda() * torch.sqrt(npower) + x

def wgn_uniform(x, snr):
    "加随机噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.rand(x.size()).cuda() * torch.sqrt(npower) + x

class AddNoiseTransform:
    def __init__(self, snr):
        self.snr = snr

    def __call__(self, x):
        self.snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/(x.size(0)*x.size(1))
        npower = xpower / self.snr
        return torch.rand(x.size()) * torch.sqrt(npower) + x
    

def select_validation_samples(data_all,label_all, num_samples):
    
    unique_labels = np.unique(label_all)
    indices_to_keep = []

    for label in unique_labels:
                    
        indices = np.where(label_all == label)[0]
        if len(indices) > num_samples:
            chosen_indices = indices[:num_samples]
        else:
            chosen_indices = indices
        indices_to_keep.extend(chosen_indices)


    return data_all[indices_to_keep], label_all[indices_to_keep]