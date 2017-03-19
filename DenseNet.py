import numpy as np
import random
from structure import Graph

class DenseNet:
#no. of units in each layer, loss fn and activation fn stored
	def __init__(self, input_dim, optim_config, loss_fn):
		self.net=Graph(input_dim, optim_config, loss_fn)
		np.random.seed(42)
		
		
	def addlayer(self, activation, units):
		self.net.addgate(activation, units)

	def train(self, X, Y):
		if self.match_dim(X,Y)==False:
			print "Dimensions of X and Y don't match"
			return -1

		if X.ndim==1:						# Added to deal with single vector inputs in case of pure SGD  
			X = np.reshape(X,(1,len(X)))
			Y = np.reshape(Y,(1,len(Y)))

		Y_pred = self.net.forward(X)
		loss_value = self.net.backward(Y)
		self.net.update()
		return loss_value

	def match_dim(self, X, Y):
		return X.ndim==Y.ndim

	def predict(self, X):
		predicted_value = self.net.forward(X)
		return predicted_value


