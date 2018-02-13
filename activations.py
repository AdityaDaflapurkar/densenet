import numpy as np
import random


class Linear:
	# Example class for the Linear layer
	def __init__(self, d, m):
		self.w = 2*np.random.random((d,m)) - 1
		self.delta=np.array([])
		self.input=np.array([])
		self.prev_delta_w=0

	def forward(self, input):
		input_with_bias=np.insert(input,0,1,axis=1)
		self.input=input_with_bias
		return np.dot(input_with_bias,self.w)

	def backward(self, dz, last):
		if last==False:
			self.delta=dz[:,1:]
			return np.dot(dz[:,1:],self.w.T)
		else: 
			self.delta=dz
			return np.dot(dz,self.w.T)

		
class ReLU:
	# Example class for the ReLU layer
	def __init__(self):
		self.input=np.array([])
		
	def forward(self, inp):
		self.input=inp
		return np.maximum(0,inp)
		
	def backward(self, dz,last):
		if last==False:
			self.input=np.insert(self.input,0,1,axis=1)
		return 1.*((self.input)>0)*dz


class Sigmoid:
	# Example class for the Sigmoid layer
	def __init__(self):
		self.output=np.array([])

	def forward(self, inp):
		res=1/(1+np.exp(-inp))
		self.output=res        
		return res

	def backward(self, dz,last):
		s = self.output
		if last==False:
			s=np.insert(s,0,1,axis=1)
		return dz * s * (1-s)


class Softmax:
	# Example class for the Softmax layer
	def __init__(self):
		self.tr_siz=0
		self.output=np.array([])

	def forward(self, input):
		self.tr_siz=len(input[0])
		s=np.sum(np.exp(input),axis=1)
		self.output = np.exp(input)/np.reshape(s,(len(s),1))
		return self.output

	def backward(self, dz, last):
		der=np.empty((len(self.output),len(self.output[0]),len(self.output[0])))
		for k in xrange(len(self.output)):
			for i in xrange(len(self.output[0])):
				for j in xrange(len(self.output[0])):
					der[k][i][j]=self.output[k][i]*(self.kronecker_delta(i, j)-self.output[k][j])
		
		res=[]
		for k in xrange(len(self.output)):
			q=np.dot(der[k],dz[k].T)
			res.append(q)
		return np.array(res)

	def kronecker_delta(self, i, j):
		return int(i==j) 
