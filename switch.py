# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:56:15 2021

@author: murra
"""
import torch
import matplotlib.pyplot as plt
from model import ctrnn

time = 400
inputs = torch.zeros(time, 1, 1)
W = 4.5*torch.ones(1,1)
b = -2.25*torch.ones(1,1)
tau = torch.ones(1,1)

initial = 0*torch.ones(1, 1)
model = ctrnn(1, 0.1, initial, time)
model.assignWeight(W, b, tau)
states_0 = torch.squeeze(model(inputs)).detach().numpy()

initial = 2*torch.ones(1, 1)
model = ctrnn(1, 0.1, initial, time)
model.assignWeight(W, b, tau)
states_2 = torch.squeeze(model(inputs)).detach().numpy()

initial = 2.25*torch.ones(1, 1)
model = ctrnn(1, 0.1, initial, time)
model.assignWeight(W, b, tau)
states_225 = torch.squeeze(model(inputs)).detach().numpy()

initial = 3*torch.ones(1, 1)
model = ctrnn(1, 0.1, initial, time)
model.assignWeight(W, b, tau)
states_3 = torch.squeeze(model(inputs)).detach().numpy()

initial = 5*torch.ones(1, 1)
model = ctrnn(1, 0.1, initial, time)
model.assignWeight(W, b, tau)
states_5 = torch.squeeze(model(inputs)).detach().numpy()

fig, ax = plt.subplots()
ax.plot(states_0,label='x_0 = 0')
ax.plot(states_2,label='x_0 = 2')
ax.plot(states_225,label='x_0 = 2.25')
ax.plot(states_3,label='x_0 = 3')
ax.plot(states_5,label='x_0 = 5')
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.set_title('Trajectories in time for 1 neuron CTRNN')
plt.show()



# if __name__ == "__main__":
