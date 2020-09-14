import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math, os
from utils import*

import time
import matplotlib.pyplot as plt
import copy
import pdb

from torch import nn
import torch
import numpy as np
from scipy.stats import beta

def set_init(block):
    for b in block:
        for layer in b:
            if type(layer) == nn.Linear:
                nn.init.normal_(layer.weight, mean=0., std=0.1)
                nn.init.constant_(layer.bias, 0.)

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
                  for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r- np.zeros_like(r) 