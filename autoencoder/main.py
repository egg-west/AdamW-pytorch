
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import AutoEncoder 
from adamW import AdamW
input_size = 28*28
batch_size = 128
EP = 10
LR = 0.005
WD = 0.1

train_set = datasets.MNIST(root='./data',
							train = True,
							transform = transforms.ToTensor(),
							download = True)
test_set = datasets.MNIST(root='./data',
							train = False,
							transform = transforms.ToTensor(),
							download = True)
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)


model = AutoEncoder(input_size = input_size)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD, amsgrad=False)# write so many arguments to get a clear view
criterion = nn.MSELoss()

losses_adam = []
for i in range(EP):
	loss_ep = 0
	idx = 0
	for batch_idx, (inputs, targets) in enumerate(train_loader):# Notice that we will not use target here.
		inputs = inputs.view([-1, 28*28]).cuda()

		optimizer.zero_grad()
		y = model(inputs)
		loss = criterion(y, inputs)
		loss.backward()
		optimizer.step()
		loss_ep += loss.item()
		idx = batch_idx
	print('EP: #{} average_loss: {}'.format(i, loss_ep/(idx+1) ))
	losses_adam.append(loss_ep)

# model.save_model()

model = AutoEncoder(input_size = input_size)
model.cuda()
optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD, amsgrad=False)# write so many arguments to get a clear view
criterion = nn.MSELoss()

losses_W = []


# wd = 0.1
for i in range(EP):
	loss_ep = 0
	idx = 0
	for batch_idx, (inputs, targets) in enumerate(train_loader):# Notice that we will not use target here.
		inputs = inputs.view([-1, 28*28]).cuda()

		optimizer.zero_grad()
		y = model(inputs)
		loss = criterion(y, inputs)
		loss.backward()

		# for group in optimizer.param_groups:
		# 	for param in group['params']:
		# 		param.data = param.data.add(-wd * LR, param.data)

		optimizer.step()
		loss_ep += loss.item()
		idx = batch_idx
	print('EP: #{} average_loss: {}'.format(i, loss_ep/(idx+1) ))
	losses_W.append(loss_ep)


plt.plot(losses_adam, label='adam')
plt.plot(losses_W, label = 'adamW')
plt.legend(loc='best')
plt.savefig('adam_adamW.jpg')


# for batch_idx, (inputs, targets) in enumerate(trainloader):
