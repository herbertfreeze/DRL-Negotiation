import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math, os
from utils import*
# from continuous_gameplay import*
from shared_adam import SharedAdam
from itertools import product
from multiprocessing import Pool
from functions import*

import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import copy
import pdb

torch.manual_seed(9)
num_episodes = 501
V = torch.Tensor([3,2,1])
T = 20
c = 1.5
r = 0.0
discount = 1.0
gamma = 0.98
a = bouwer(V,T,c,r)
Net = receive_net_ANN()
optimizer = optim.Adam(Net.parameters(),
                           lr=0.0001)
losses = np.zeros(num_episodes)
start = time.time()
times = []
loss = 0

for ep in range(num_episodes):
    if ep % 500 == 0:
        print("Epoch ", ep, "  with loss at ", loss)
        print("Episode Time is: ", time.time() - start)
        start = time.time()
        output_training_metrics(ep,Net,a)
    states  = []
    actions = []
    rewards = []

    t = 0
    accepted = False
    X = torch.ones(3*T+1) * -100

    while not accepted and t < 20:
        X[3*t:3*t+3] = 1 - a.generate_offer(t)[:3]
        X[-1] = t
        your_offer = torch.cat((1 - a.generate_offer(t)[:3], torch.Tensor([t])))
        states.append(your_offer)
#         states.append(X.clone())
#         accepted = Net.choose_action(X.unsqueeze(0))
        accepted = Net.choose_action(your_offer.unsqueeze(0))
        actions.append(accepted)
        t += 1

    t_final = t - 1
    times.append(t_final)

    if t == 20:
        final_reward = torch.Tensor([-1])
    else:
#         final_reward = torch.sum(X[3*t_final:3*t_final + 3] * Net.V) * discount**t_final
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
    #     print(c_loss, a_loss)
        loss = (c_loss + a_loss).mean()
        losses[ep] = loss

        optimizer.zero_grad()
        a_loss.mean().backward(retain_graph = True)
        c_loss.mean().backward()
    #     loss.backward()
        optimizer.step()

np.save("c{}-d{}-losses".format(c,discount), losses)
np.save("c{}-d{}-times".format(c,discount),  times)
