# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:43:00 2021

@author: murra
"""
import torch
import matplotlib.pyplot as plt
from model import ctrnn

time = 500
inputs = torch.zeros(time, 1, 2)
dim = 2
dt = 0.1
W = torch.tensor([[4.5, -1], [1, 4.5]]).T
b = torch.tensor([[-2.75, -1.75]])
tau = torch.ones(1,2)

initial = torch.tensor([[0, 0]])
model = ctrnn(dim, dt, initial, time)
model.assignWeight(W, b, tau)
states_0 = torch.squeeze(model(inputs)).detach().numpy()

initial = torch.tensor([[1, 1]])
model = ctrnn(dim, dt, initial, time)
model.assignWeight(W, b, tau)
states_2 = torch.squeeze(model(inputs)).detach().numpy()

initial = torch.tensor([[0.6, 0.6]])
model = ctrnn(dim, dt, initial, time)
model.assignWeight(W, b, tau)
states_225 = torch.squeeze(model(inputs)).detach().numpy()

initial = torch.tensor([[0.4, 0.4]])
model = ctrnn(dim, dt, initial, time)
model.assignWeight(W, b, tau)
states_3 = torch.squeeze(model(inputs)).detach().numpy()

initial = torch.tensor([[0, 1]])
model = ctrnn(dim, dt, initial, time)
model.assignWeight(W, b, tau)
states_5 = torch.squeeze(model(inputs)).detach().numpy()

fig, ax = plt.subplots()
ax.plot(states_0[:,0], states_0[:,1], label='x_0 = 0')
ax.plot(states_2[:,0], states_2[:,1], label='x_0 = 2')
ax.plot(states_225[:,0], states_225[:,1], label='x_0 = 2.25')
ax.plot(states_3[:,0], states_3[:,1], label='x_0 = 3')
ax.plot(states_5[:,0], states_5[:,1], label='x_0 = 5')
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_title('Trajectories in 2D space for 2 neuron CTRNN')
plt.show()
