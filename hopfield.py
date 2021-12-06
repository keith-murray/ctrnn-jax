# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 07:12:12 2021

@author: murra
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ctrnn

def constructWeights(dim, seqs):
    W = torch.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                W[i,j] = 0
            else:
                for seq in seqs:
                    W[i,j] += seq[i]*seq[j]
    return W

time = 10000
dim = 100
dt = 0.1
inputs = torch.zeros(time, 1, dim)

img1 = [1,1,0,0,1,1,0,0,1,1,
        1,1,0,0,1,1,0,0,1,1,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        1,1,0,0,1,1,0,0,1,1,
        1,1,0,0,1,1,0,0,1,1]

img2 = [0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,1,1,0,0,1,1,0,0,
        0,0,1,1,0,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,1,0,0,
        0,0,1,1,0,0,1,1,0,0,
        0,0,0,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,0]

seq1 = 2*torch.tensor(img1)-1
seq2 = 2*torch.tensor(img2)-1

W = constructWeights(dim, [seq1, seq2])
b = 0*torch.ones(1, dim)
tau = torch.ones(1, dim)

init = [0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0]

init = [0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,1,0,0,1,0,0,0,
        0,0,0,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,0]

initial = torch.tensor(init)
model = ctrnn(dim, dt, initial, time)
model.assignWeight(W, b, tau)
states0 = torch.squeeze(model(inputs))
states = torch.tensor([round(x.item()) for x in states0[-1,:]])

plt.imshow(np.reshape(states.detach().numpy(), (10,10)))
