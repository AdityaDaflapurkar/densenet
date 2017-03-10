import numpy as np
import random

class DenseNet:
#no. of units in each layer, loss fn and activation fn stored
    def __init__(self, input_dim, optim_config, loss_fn):
        """
        Initialize the computational graph object.
        """
        pass
        
    def addlayer(self, activation, units):
        """
        Modify the computational graph object by adding a layer of the specified type.
        """
        pass

    def train(self, X, Y):
        """
        This train is for one iteration. It accepts a batch of input vectors.
        It is expected of the user to call this function for multiple iterations.
        """
        return loss_value

    def predict(self, X):
        """
        Return the predicted value for all the vectors in X.
        """
return predicted_value


class Graph:
#
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
			layer=Linear(current_layer_size+1,units)
			self.layers.append(layer)
			current_layer_size=units

		else if activation==2:
			layer=Linear(current_layer_size+1,units)
			self.layers.append(layer)
			layer=Relu()
			self.layers.append(layer)
			current_layer_size=units

		else if activation==3:
			layer=Linear(current_layer_size+1,units)
			self.layers.append(layer)
			layer=Sigmoid()
			self.layers.append(layer)
			current_layer_size=units

		else if activation==4:
			layer=Softmax()
			self.layers.append(layer)
			current_layer_size=units

		else:
			print "Invalid Gate ID!!"



    def forward(self, inp):

		for i in xrange(len(self.layers)):
			predicted_value=self.layer[i].forward(inp)
			inp=predicted_value
		
		self.output=predicted_value
		return predicted_value



    def backward(self, expected):
		delta=np.array([])
		if self.lf==1:
			loss=L1_loss()
			delta=loss.backward(expected,self.output)

		else if self.lf==2:
			loss=L2_loss()
			delta=loss.backward(expected,self.output)

		else if self.lf==3:
			loss=Cross_Entropy()
			delta=loss.backward(expected,self.output)
		
		else if self.lf==4:			
			loss=SVM_loss()
			delta=loss.backward(expected,self.output)

		else:
			print "Invalid Loss Function ID!!"
		
		
		
		for i in reversed(xrange(len(self.layers))):
			loss_val=self.layer[i].backward(delta)
			delta=predicted_value
    
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

class Error:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
    def __init__(self):
        pass

    def forward(self, yd, yp):
		return yd-yp
    def backward(self):
		return -1

class L1_loss:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
    def __init__(self):
        pass

    def forward(self, error):
		return np.sum(abs(error))

    def backward(self, error):	
		return np.nan_to_num(abs(error)/error)


class L2_loss:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
    def __init__(self):
        pass

    def forward(self, input):
		return ((np.sum(error))**2)/2
    def backward(self, error):
		return error


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
		np.sum(np.maximum(0,1-yd*yp)

    def backward(self, yd, x):
		yp=np.dot(self.w.T, x)
		if yd*yp<1:				
			return -yd*x
		else:
			return 0
