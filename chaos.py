# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:15:11 2021

@author: murra
"""
import torch
import matplotlib.pyplot as plt
from model import ctrnn

time = 5000
inputs = torch.zeros(time, 1, 3)
dim = 3
dt = 0.1
W = torch.tensor([[5.422, -0.24, 0.535], 
                  [-0.018, 4.59, -2.25],
                  [2.750, 1.210, 3.885]]).T
b = torch.tensor([[-4.108, -2.787, -1.114]])
tau = torch.tensor([[1, 2.5, 1]])

initial = torch.tensor([[0.5, 0.5, 0.5]])
model = ctrnn(dim, dt, initial, time)
model.assignWeight(W, b, tau)
states = torch.squeeze(model(inputs)).detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2])
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_zlabel('x_2')
ax.set_title('Trajectories in 3D space for 3 neuron CTRNN')
plt.show()

fig, ax = plt.subplots()
ax.plot(states[:,0], label='x_0')
ax.plot(states[:,1], label='x_1')
ax.plot(states[:,2], label='x_2')
ax.set_title('Trajectories in time for 3 neuron CTRNN')
ax.legend(loc=(0.8, 0.5))
plt.show()