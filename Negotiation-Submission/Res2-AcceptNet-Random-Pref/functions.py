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
import matplotlib
import copy
import pdb

def sample_P_from_Plane(ut,dim = 41):
    xspace = np.linspace(0,1,dim)
    yspace = np.linspace(0,1,dim)
    zspace = np.linspace(0,1,dim)
    ospace = np.zeros((dim*dim*dim,3))
    i = 0
    for x in xspace:
        for y in yspace:
            for z in zspace:
                ospace[i] = np.array([x,y,z])
                i += 1
    uspace = np.sum(ospace*np.array([3,2,1]),axis=1)
    valid = ospace[(np.abs(uspace-ut) < 0.01)]
    val_ind = np.random.choice(len(valid))
    return torch.Tensor(valid[val_ind])

def calc_action_space(ut,dim = 51):
    xspace = np.linspace(0,1,dim)
    yspace = np.linspace(0,1,dim)
    zspace = np.linspace(0,1,dim)
    ospace = np.zeros((dim*dim*dim,3))
    i = 0
    for x in xspace:
        for y in yspace:
            for z in zspace:
                ospace[i] = np.array([x,y,z])
                i += 1
    uspace = np.sum(ospace*np.array([3,2,1]),axis=1)
    valid = ospace[(np.abs(uspace-ut) < 0.02)]
    return valid

def boulware_actions(c):
    r = 0.0
    T = 20
    V = torch.Tensor([3,2,1])
    c_list = [c]
    X = []
    P_res = []
    t = 0
    for c in c_list:
        a = boulware(V,T,c,r)
        for t in range(T+1):
            state = a.generate_offer(t) # what the agent sees
            uts = a.calc_decision_util(t)
            X.append(state)
            P_res.append(torch.Tensor([uts*6]))

    P_res = torch.stack(P_res)
    X1 = torch.stack(X)
    return X1, P_res

class boulware(object):
    def __init__(self,V,T, c,r):
        self.V = V
        self.T = T
        self.c = c
        self.r = r
        self.gen_action_space()
    
    def gen_action_space(self):
        ut_list = []
        for t in range(self.T+1):
            uts = self.calc_decision_util(t)
            uts *= 6
            ut_list.append(uts)
            
        space_dict = {}
        for t,u  in enumerate(ut_list):
            space_dict[t] = calc_action_space(u)
            if len(space_dict[t]) == 0:
                pdb.set_trace()
        self.space_dict = space_dict
        print("Created Action Space")
                
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
#         m = torch.distributions.Normal(0,0.10)
        
#         X = torch.clamp((ut / self.V[0] + m.sample()),0,1)
#         if X == 1:
#             ut -= X*self.V[0]
#             Y = torch.clamp((ut / self.V[1] + m.sample()),0,1)
#             if Y == 1:
#                 ut -= Y*self.V[1]
#                 Z = torch.clamp((ut / self.V[2]+ m.sample()),0,1)
#             else:
#                 Z = 0
#         else:
#             Y = Z = 0
#         return torch.Tensor([X,Y,Z,t])

#         offer = sample_P_from_Plane(ut)
        valid = self.space_dict[int(t)]
        val_ind = np.random.choice(len(valid))
        offer = torch.Tensor(valid[val_ind])
        return torch.cat((offer,torch.Tensor([t])))

    def receive(self,offer,t):
        my_offer = 1-offer
        ut = self.calc_decision_util(t)
        ut *= torch.sum(self.V)
        if torch.sum(self.V * my_offer) > ut:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])

X_DIM  = 4
H1_DIM = 128
H2_DIM = 256
REC_DIM = 2
V_DIM = 1
torch.manual_seed(8)

class receive_net_ANN(nn.Module):
    def __init__(self):
        super(receive_net_ANN, self).__init__()

        self.V = torch.Tensor([1,2,3])
        self.name = "RECEIVE NET"

        self.base = nn.Sequential(
            nn.Linear(X_DIM,H1_DIM),
            nn.ReLU6(),
            nn.Linear(H1_DIM, H2_DIM),
            nn.ReLU6(),
        )

        self.receive = nn.Sequential(
            nn.Linear(H2_DIM, H2_DIM),
            nn.ReLU6(),
            nn.Linear(H2_DIM, REC_DIM),
            nn.Softplus()
        )

        self.m_rec = torch.distributions.Categorical

        self.value = nn.Sequential(
            nn.Linear(H2_DIM, H2_DIM),
            nn.ReLU6(),
            nn.Linear(H2_DIM, H2_DIM),
            nn.ReLU6(),
            nn.Linear(H2_DIM, V_DIM),
        )

        set_init([self.base, self.receive, self.value])

    def forward(self, x):
        out  = self.base(x)
        logits = self.receive(out)
        return logits, self.value(out)

    def choose_action(self,x,show=False):
        logits, __ = self.forward(x)
        ### Choose accept or no ###
        ps = logits + torch.Tensor([[0,0]])
        prob = F.softmax(ps, dim=1)
        rec_distrib = self.m_rec(prob)
        receive = rec_distrib.sample()
        return receive

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

    
def plot_training(c,discount):
    path = "c{}-d{}".format(c,discount)
    if not os.path.exists(path + "/figures"):
        os.mkdir(path+ "/figures")
    losses  = np.load(path+"/c{}-d{}-losses.npy".format(c,discount))
    rewards_aggregate = np.load(path+"/c{}-d{}-rewards.npy".format(c,discount))
    times = np.load(path+"/c{}-d{}-times.npy".format(c,discount))
    plt.figure(figsize=(12,8))
    matplotlib.rc('xtick', labelsize=22) 
    matplotlib.rc('ytick', labelsize=22)
    plt.plot(losses[:], label="Loss")
    plt.plot(rewards_aggregate, label="Rewards")
    plt.plot(times, label =  "Playing Time")
    plt.legend(fontsize = 15)
    
def plot_agent(c,discount,ep):
    path = "c{}-d{}".format(c,discount)
    logit_0 = np.load(path + "/data/{}-logit0.npy".format(ep))
    logit_1 = np.load(path + "/data/{}-logit1.npy".format(ep))
    acceptance_prob = np.load(path + "/data/{}-accept_prob.npy".format(ep))
    binomial_probs  = np.load(path + "/data/{}-binom_prob.npy".format(ep))
    vals_list = np.load(path + "/data/{}-val_function.npy".format(ep))
    
    plt.figure(figsize=(9,12))
    
    plt.subplot(2,1,1)
    plt.plot(acceptance_prob, label = "Acceptance Prob | Timestep")
    plt.plot(binomial_probs, label = "Binomial Probabilities")
    plt.legend(fontsize = 15)
    
    plt.subplot(2,1,2)
    plt.plot(vals_list, label = "Value Function")
    plt.plot(logit_0, label = "Logit 0")
    plt.plot(logit_1, label = "Logit 1")
    plt.legend(fontsize = 15)
    
def train(num_episodes,c,discount, LR =0.0001 ):
    torch.manual_seed(9)
    V = torch.Tensor([3,2,1])
    T = 20
    r = 0.0
    gamma = 0.98
    a = boulware(V,T,c,r)
    Net = receive_net_ANN()
    optimizer = optim.Adam(Net.parameters(),
                               LR)
    losses = np.zeros(num_episodes)
    rewards_aggregate = np.zeros(num_episodes)
    times = np.zeros(num_episodes)
    start = time.time()
    loss = 0

    path = "c{}-d{}".format(c,discount)
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path+"/data")

    for ep in range(num_episodes):
        if ep % 500 == 0:
            print("Epoch ", ep, "  with loss at ", loss)
            print("Episode Time is: ", time.time() - start)
            start = time.time()
            output_training_metrics(ep,Net,a,path)
        states  = []
        actions = []
        rewards = []

        t = 0
        accepted = False
        while not accepted and t < 20:
            your_offer = torch.cat((1 - a.generate_offer(t)[:3], torch.Tensor([t])))
            states.append(your_offer)
            accepted = Net.choose_action(your_offer.unsqueeze(0))
            actions.append(accepted)
            t += 1

        t_final = t - 1
        times[ep] = t_final

        if t == 20:
            final_reward = torch.Tensor([-1])
        else:
            final_reward = torch.sum(your_offer[:3] * Net.V) * discount**t_final

        for i in range(t):
            r = gamma**(t_final-i) * final_reward
            rewards.append(r)

        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions).squeeze()
        rewards_tensor = torch.stack(rewards)

        for __ in range(10):
            logits, vals = Net(states_tensor)
            # Critic Loss
            td = rewards_tensor - vals.squeeze()
            c_loss = td.pow(2)

            ## Receive Loss
            probs = F.softmax(logits, dim=1)
            m = Net.m_rec(probs)
            exp_v = m.log_prob(actions_tensor).squeeze() * td.detach()
            a_loss = -exp_v

            loss = (c_loss + a_loss).mean()
            losses[ep] = loss
            rewards_aggregate[ep] = final_reward

            optimizer.zero_grad()
            a_loss.mean().backward(retain_graph = True)
            c_loss.mean().backward()
            optimizer.step()

    np.save(path + "/c{}-d{}-losses".format(c,discount), losses)
    np.save(path + "/c{}-d{}-times".format(c,discount),  times)
    np.save(path + "/c{}-d{}-rewards".format(c,discount),  rewards_aggregate)
    torch.save(Net.state_dict(),path + ".th")
    return Net