import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math, os
from utils import*

from itertools import product
from multiprocessing import Pool
from functions import*

import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import copy
import pdb

class boulware(object):
    def __init__(self,V,T, c,r):
        self.V = V
        self.T = T
        self.c = c
        self.r = r
    
    def calc_Ft(self,t):
        return (t/self.T)**(1/self.c)

    def calc_decision_util(self,t):
        Ft = self.calc_Ft(t)
        ut = self.r + (1 - self.r)*(1-Ft)
        return ut
    
    def generate_offer(self,t):
        # How to decide which axis to concede.
        ut = self.calc_decision_util(t)
        ut *= torch.sum(self.V)
        
        X = torch.clamp((ut / self.V[0]) + np.random.normal(scale=0.05) ,0,1)
        if X == 1:
            ut -= X*self.V[0]
            Y = torch.clamp((ut / self.V[1])+ np.random.normal(scale=0.05) ,0,1)
            if Y == 1:
                ut -= Y*self.V[1]
                Z = torch.clamp((ut / self.V[2])+ np.random.normal(scale=0.05) ,0,1)
            else:
                Z = 0
        else:
            Y = Z = 0
        
#         print(torch.Tensor([X,Y,Z]))
        return torch.Tensor([X,Y,Z,t])

    def receive(self,offer,t):
        my_offer = 1-offer
        ut = self.calc_decision_util(t)
        ut *= torch.sum(self.V)
        if torch.sum(self.V * my_offer) > ut:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])

X_DIM  = 4
H1_DIM = 256
H2_DIM = 512
REC_DIM = 2
OFF_DIM = 1
V_DIM = 1
LR = 0.0001
EPISODES = 100000
DEADLINE = 20
DISCOUNTS = list(np.linspace(0,1,21))
torch.manual_seed(7)

def calc_rewards(P_res, actions, own_v, opp_v):
    receiving = 1 - actions
    own_ut = torch.sum(actions   * own_v, dim=1,keepdim=True)   
    opp_ut = torch.sum(receiving * opp_v, dim=1,keepdim=True)   
    rewards = own_ut * (P_res < opp_ut).float()
    return rewards

    
def output_training_metrics(ep,Net,a,path):
    ## Extract acceptance probabilities, logits
    logit_0 = np.zeros(20)
    logit_1 = np.zeros(20)
    acceptance_prob = np.zeros(20)
    vals_list = np.zeros(20)

    for t in range(20):
        X = torch.cat((1 - a.generate_offer(t)[:3], torch.Tensor([t])))
        accepted = Net.choose_action(X.unsqueeze(0))
        logits, vals = Net(X.unsqueeze(0))
        logits = logits.squeeze()
        acceptance_prob[t] = logits[1]/torch.sum(logits)
        vals_list[t] = vals.squeeze().detach().numpy()
        logit_0[t], logit_1[t] = logits[0], logits[1]

    # Construct binomial distribution
    reject_cum_prob = 1
    timestep = np.arange(20)
    binomial_probs = np.zeros(20)
    for t in timestep:
        accept_prob   = reject_cum_prob * acceptance_prob[t]
        reject_cum_prob *= 1-acceptance_prob[t]
        binomial_probs[int(t)] = accept_prob

    np.save(path + "/data/{}-logit0".format(ep), logit_0)
    np.save(path + "/data/{}-logit1".format(ep), logit_1)
    np.save(path + "/data/{}-accept_prob".format(ep), acceptance_prob)
    np.save(path + "/data/{}-binom_prob".format(ep), binomial_probs)
    np.save(path + "/data/{}-val_function".format(ep), vals_list)
