import numpy as np
import random


class L1_loss:
	# Example class for the L1 loss
	def __init__(self):
		pass

	def forward(self, yd, yp):
		return np.mean(abs(np.mean((yd-yp),axis=1)))

	def backward(self, yd, yp):	
		return -np.nan_to_num(abs(yd-yp)/(yd-yp))


class L2_loss:
	# Example class for the L2 loss
	def __init__(self):
		pass

	def forward(self, yd, yp):
		return np.mean((np.sum((yp-yd)**2,axis=1)))
	
	def backward(self, yd, yp):
		return 2*(yp-yd)


class Cross_Entropy:
	# Example class for the Cross Entropy loss
	def __init__(self):
		pass

	def forward(self, yd, yp):
		return  np.mean(-np.sum(yd*np.log(yp),axis=1))

	def backward(self, yd, yp):
		return -yd/yp


class SVM_loss:
	# Example class for the SVM loss
	def __init__(self,m):
		self.margin=m
		self.current_loss=[]

	def forward(self, yd, yp):
		class_index=np.where(yd==1)[1]
		loss=[]
		
		for i in xrange(len(class_index)):
			mask=np.ones(len(yd[0]))
			mask[class_index[i]]=0
			current=np.maximum(0,yp[i]-yp[i][class_index[i]]+self.margin)*mask
			loss.append(current)
		self.current_loss=loss
		final_loss=np.sum(self.current_loss,axis=1)
		
		
		return np.mean(final_loss)	
		
	def backward(self, yd):
		class_index=np.where(yd==1)[1]
		grad=np.zeros((len(yd),len(yd[0])))
		for i in xrange(len(yd)):
			for j in xrange(len(yd[0])):
				if  j==class_index[i]:
					grad[i][j]=-np.sum(1.*(self.current_loss[i]>0))
				else:
					grad[i][j]=1.*(self.current_loss[i][j]>0)
		return grad
