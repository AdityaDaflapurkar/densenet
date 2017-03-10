import numpy as np
import random

class DenseNet:
#no. of units in each layer, loss fn and activation fn stored
	def __init__(self, input_dim, optim_config, loss_fn):
		self.net=Graph(input_dim, optim_config, loss_fn)
	
		
		
	def addlayer(self, activation, units):
		self.net.addgate(activation, units)

	def train(self, X, Y):
		Y_pred = self.net.forward(X)
		loss_value = self.net.backward(Y)
		return loss_value

	def predict(self, X):
		predicted_value = self.net.forward(X)
		return predicted_value


class Graph:

	# Computational graph class
	def __init__(self, input_dim, optim_config, loss_fn):
		self.id = input_dim
		self.oc = optim_config
		self.lf = loss_fn
		self.eta = 0.1
		self.layers = []
		self.current_layer_size = input_dim
		self.output = []
	

	def addgate(self, activation, units=0):

		if activation==1:
			layer=Linear(self.current_layer_size+1,units)
			self.layers.append(layer)
			self.current_layer_size=units

		elif activation==2:
			layer=Linear(self.current_layer_size+1,units)
			self.layers.append(layer)
			layer=Relu()
			self.layers.append(layer)
			self.current_layer_size=units
	
		elif activation==3:
			layer=Linear(self.current_layer_size+1,units)
			self.layers.append(layer)
			layer=Sigmoid()
			self.layers.append(layer)
			self.current_layer_size=units
	
		elif activation==4:
			layer=Softmax()
			self.layers.append(layer)
			self.current_layer_size=units
	
		else:
			print "Invalid Gate ID!!"



	def forward(self, inp):

		for i in xrange(len(self.layers)):
			predicted_value=self.layers[i].forward(inp)
			inp=predicted_value
		
		self.output=predicted_value
		return predicted_value



	def backward(self, expected):
		delta=np.array([])
		if self.lf==1:
			loss=L1_loss()
			delta=loss.backward(expected,self.output)

		elif self.lf==2:
			loss=L2_loss()
			delta=loss.backward(expected,self.output)

		elif self.lf==3:
			loss=Cross_Entropy()
			delta=loss.backward(expected,self.output)
		
		elif self.lf==4:			
			loss=SVM_loss()
			delta=loss.backward(expected,self.output)

		else:
			print "Invalid Loss Function ID!!"
		
		
		
		for i in reversed(xrange(len(self.layers))):
			loss_val=self.layer[i].backward(delta)
			delta=loss_val
	
		return loss_val



	def update(self):
		for i in xrange(len(self.layers)):
			if  isinstance(self.layers[i], Linear):
				self.layers[i].w=w+(self.eta)*(self.layers[i].input)*(self.layers[i].delta)



class ReLU:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.


	def __init__(self):
		self.input=np.array([])
		
	def forward(self, inp):
		self.input=inp
		return np.maximum(0,inp)
		
	def backward(self, dz):
		return  1.*((self.input)>0)*dz




class Softmax:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def __init__(self):
		pass

	def forward(self, input):
		return np.exp(input)/np.sum(np.exp(input))

	def backward(self, dz):
		res=np.empty(len(dz),len(dz))
		for i in xrange(len(dz)):
			for j in xrange(len(dz)):
				res[i][j]=dz[i]*(self.kronecker_delta(i, j)-dz[j])	
		return res

	def kronecker_delta(self, i, j):
		return int(i==j) 

class Sigmoid:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def __init__(self):
		self.output=np.array([])

	def forward(self, inp):
		res=1/(1+np.exp(-inp))
		self.output=res        
		return res

	def backward(self, dz):
		s = self.output
		return dz * s * (1-s)

class Linear:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def __init__(self, d, m):
		self.w = 2*np.random.random((d,m)) - 1
		self.delta=np.array([])
		self.input=np.array([])

	def forward(self, input):
		input_with_bias=np.insert(input,0,1,axis=1)
		self.input=input_with_bias
		return np.dot(input_with_bias.T,self.w)

	def backward(self, dz):
		self.delta=dz
		return np.dot(dz,self.w.T)


class L1_loss:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def __init__(self):
		pass

	def forward(self, yd, yp):
		return np.sum(abs(yd-yp))

	def backward(self, yd, yp):	
		return -np.nan_to_num(abs(yd-yp)/(yd-yp))


class L2_loss:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def __init__(self):
		pass

	def forward(self, yd, yp):
		return ((np.sum(yd-yp))**2)/2
	def backward(self, yd, yp):
		return -(yd-yp)


class Cross_Entropy:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def __init__(self):
		pass

	def forward(self, yd, yp):
		return -np.sum(yd*np.log(yp)+(1-yd)*np.log(1-yp))

	def backward(self, yd, yp):
		return (yp-yd)/(yp*(1-yp))

class SVM_loss:
	# Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def __init__(self, wt):
		self.w=wt

	def forward(self, yd, x):
		yp=np.dot(self.w.T, x)
		np.sum(np.maximum(0,1-yd*yp))

	def backward(self, yd, x):
		yp=np.dot(self.w.T, x)
		if yd*yp<1:				
			return -yd*x
		else:
			return 0

if __name__ == "__main__":
	random.seed(1)

	
	nn_model = DenseNet(2,3,2)

	# Create layer 2 (a single neuron with 4 inputs)
	nn_model.addlayer(3,4)
	nn_model.addlayer(3,1)
	x = np.array([  [1,0,1],
					[1,1,0],
					[1,1,1],
					[1,0,0]  ])

	y = np.array([  [1],
					[1],
					[0],
					[0]  ])

	for i in xrange(1000):
		for j in xrange(4):
			nn_model.train(x[j],y[j])
	
	

