import os
import torch
import torch.nn as nn
import torch.nn.functional as functional


class Encoder(nn.Module):
	def __init__(self, input_size, bottle_size = 2):
		super(Encoder, self).__init__()
		self.ln1 = torch.nn.Linear(input_size, 128)
		self.ln2 = torch.nn.Linear(128, 64)
		self.ln3 = torch.nn.Linear(64, 12)
		self.ln4 = torch.nn.Linear(12, bottle_size)

		self.relu = nn.ReLU()

	def forward(self, input):
		y_ = self.relu(self.ln1(input))
		y_ = self.relu(self.ln2(y_))
		y_ = self.relu(self.ln3(y_))
		y_ = self.ln4(y_)
		return y_


class Decoder(nn.Module):
	def __init__(self, input_size, bottle_size = 2):
		super(Decoder, self).__init__()
		self.ln1 = torch.nn.Linear(bottle_size, 12)
		self.ln2 = torch.nn.Linear(12, 64)
		self.ln3 = torch.nn.Linear(64, 128)
		self.ln4 = torch.nn.Linear(128, input_size)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		y_ = self.relu(self.ln1(input))
		y_ = self.relu(self.ln2(y_))
		y_ = self.relu(self.ln3(y_))
		y_ = self.sigmoid(self.ln4(y_))
		return y_


class AutoEncoder(nn.Module):
	def __init__(self ,input_size):
		super(AutoEncoder, self).__init__()
		self.encoder = Encoder(input_size)
		self.decoder = Decoder(input_size)

		# self.encoder.cuda()
		# self.decoder.cuda()


	def forward(self, input):
		return self.decoder(self.encoder(input))


	def save_model(self, path='model'):
		if not os.path.exists(path):
			os.mkdir(path)
		torch.save(
			self.encoder.state_dict(),
			'{}/encoder.pkl'.format(path))
		torch.save(
			self.encoder.state_dict(),
			'{}/decoder.pkl'.format(path))

	def load_model(self, path='model'):
		self.encoder.load_state_dict(
			torch.load('{}/encoder.pkl'.format(path)))
		self.decoder.load_state_dict(
			torch.load('{}/decoder.pkl'.format(path)))
