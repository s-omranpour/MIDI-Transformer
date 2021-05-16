import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def nucleus_sample(x, top_p=1., t=1.):
    probs = F.softmax(x.cpu().detach()/t)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=0)
    threshold = cumsum_sorted_probs > top_p
    idx = probs.size(0) - sum(threshold) + 1
    cand_indices = sorted_indices[:idx]
    cand_probs = probs[cand_indices]
    cand_probs /= cand_probs.sum()
    word = np.random.choice(cand_indices.numpy(), size=1, p=cand_probs.numpy())[0]
    return word

def simple_sample(x):
    return torch.multinomial(F.softmax(x.cpu().detach()), num_samples=1, replacement=True)
