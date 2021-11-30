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

initial = 0*torch.ones(1, 1)
model = ctrnn(1, 1, 0.1, initial, time)
model.assignWeight(W, b)
states_0 = torch.squeeze(model(inputs)).detach().numpy()

initial = 2*torch.ones(1, 1)
model = ctrnn(1, 1, 0.1, initial, time)
model.assignWeight(W, b)
states_2 = torch.squeeze(model(inputs)).detach().numpy()

initial = 2.25*torch.ones(1, 1)
model = ctrnn(1, 1, 0.1, initial, time)
model.assignWeight(W, b)
states_225 = torch.squeeze(model(inputs)).detach().numpy()

initial = 3*torch.ones(1, 1)
model = ctrnn(1, 1, 0.1, initial, time)
model.assignWeight(W, b)
states_3 = torch.squeeze(model(inputs)).detach().numpy()

initial = 5*torch.ones(1, 1)
model = ctrnn(1, 1, 0.1, initial, time)
model.assignWeight(W, b)
states_5 = torch.squeeze(model(inputs)).detach().numpy()

fig, ax = plt.subplots()
ax.plot(states_0)
ax.plot(states_2)
ax.plot(states_225)
ax.plot(states_3)
ax.plot(states_5)
ax.set_xlabel('time')
ax.set_title('Trajectories in time for 1-D CTRNN')
plt.show()



# if __name__ == "__main__":
