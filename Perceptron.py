import numpy as np
import random
class Perceptron:
	
	def __init__(self):
		self.weights=np.random.rand(1,3)


	def train(self,X,Y,iterations):		
		for i in xrange(iterations):
			for j in xrange(4):
				predicted=self.predict(X[j])
				error=Y[j]-predicted
				self.weights=self.weights+0.1*error*X[j]
		
	def predict(self,X):
		sum=self.weightedSum(X)
		return self.step(sum)

	def weightedSum(self,X):
		return np.dot(X,np.transpose(self.weights))

	
	def step(self,x):
		return np.piecewise(x,[x<0,x>=0],[0,1])
		

if __name__=='__main__':
		p=Perceptron()
		#AND function
		X=np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
		Y=np.array([0,0,0,1])

		#XOR function
		#X=np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
		#Y=np.array([0,1,1,0])

		print "Initial weight values : ",p.weights
		p.train(X,Y,100)
		print "Weight values after training : ",p.weights		
		result=map(int,p.predict(X))
		print "Training set(X) : \n",map(list,X)

		print "Predicted result : ",result
		print "Actual result(Y) : ",map(int,Y)
	
