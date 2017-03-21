import numpy as np
import random
from activations import Linear,ReLU,Sigmoid,Softmax
from losses import L1_loss,L2_loss,Cross_Entropy,SVM_loss


class Optimizer:

	def __init__(self, learning_rate, momentum_eta = 0.0):
		self.lr=learning_rate							# Learning rate
		self.eta=momentum_eta							# Momentum

	def weight_update(self, linear):
		prev_output=linear.input						# output of the layer before current linear layer
		next_delta=linear.delta							# delta of next layer to the current linear layer
		dataset_size=len(next_delta)
		curr_delta_w=self.lr*np.dot(prev_output.T,next_delta)/dataset_size
		result_delta_w=curr_delta_w+(self.eta*linear.prev_delta_w)
		linear.prev_delta_w=curr_delta_w
		return result_delta_w


class Graph:

	# Computational graph class
	def __init__(self, input_dim, optim_config, loss_fn):
		self.id = input_dim
		self.lf = loss_fn
		self.layers = []							# List containing all layer objects in the neural net
		self.current_layer_size = input_dim
		self.output = []							# Output of the neural net
		self.optimizer=optim_config

	def addgate(self, activation, units=0):

		if activation=='Linear':
			layer=Linear(self.current_layer_size+1,units)			# bias unit added here
			self.layers.append(layer)					# weight matrix added, no activation here
			self.current_layer_size=units

		elif activation=='ReLU':
			layer=Linear(self.current_layer_size+1,units)			# bias unit added here
			self.layers.append(layer)					# weight matrix added
			layer=ReLU()
			self.layers.append(layer)					# ReLU activation nodes added on weight matrix
			self.current_layer_size=units
	
		elif activation=='Sigmoid':
			layer=Linear(self.current_layer_size+1,units) 			# bias unit added here
			self.layers.append(layer)
			layer=Sigmoid()
			self.layers.append(layer)					# Sigmoid activation nodes added on weight matrix
			self.current_layer_size=units
	
		elif activation=='Softmax':
			layer=Softmax()
			self.layers.append(layer)
			self.current_layer_size=units
	
		else:
			print "Invalid Gate ID"
		


	def forward(self, inp):
		predicted_value=np.array([])
		for i in xrange(len(self.layers)):
			predicted_value=self.layers[i].forward(inp)
			inp=predicted_value
		
		self.output=predicted_value
		return predicted_value



	def backward(self, expected):
		loss_val=0
		delta=np.array([])
		if self.lf=='L1':
			loss=L1_loss()
			loss_val=loss.forward(expected,self.output)
			delta=loss.backward(expected,self.output)

		elif self.lf=='L2':
			loss=L2_loss()
			loss_val=loss.forward(expected,self.output)
			delta=loss.backward(expected,self.output)
			
		elif self.lf=='Softmax':
			loss=Cross_Entropy()
			loss_val=loss.forward(expected,self.output)
			delta=loss.backward(expected,self.output)
		
		elif self.lf=='SVM':			
			loss=SVM_loss(10)
			loss_val=loss.forward(expected,self.output)
			delta=loss.backward(expected)

		else:
			print "Invalid Loss Function ID"

		
		new_delta=np.array([])
		linear_not_found=True						# True if no linear layer encountered yet during backpropagation
		for i in reversed(xrange(len(self.layers))):
			new_delta=self.layers[i].backward(delta,linear_not_found)
			delta=new_delta
			if  isinstance(self.layers[i], Linear): 		# Linear layer encountered for the first time  
				linear_not_found=False

		return loss_val



	def update(self):
		for i in xrange(len(self.layers)):
			if  isinstance(self.layers[i], Linear):
				self.layers[i].w=(self.layers[i].w)-self.optimizer.weight_update(self.layers[i])
				#print self.layers[i].w," weights"
				


