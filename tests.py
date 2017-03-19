import numpy as np
import random
from DenseNet import DenseNet
from structure import Optimizer



testcase=raw_input()
if testcase=='1':
	#L1 loss regression test
	
	#Configuration
	opti=Optimizer(0.1,0)
	nn_model = DenseNet(2,opti,"L1")
	nn_model.addlayer('Linear',1)

	#Model Training set: y=c0+5*x1+3*x2+error
	c=np.array([[2,5,3]])
	error=np.random.uniform(-1,1,[100,1])	
	X=np.random.uniform(-1,1,[100,2])
	Xi=np.insert(X,0,1,axis=1)
	Y=np.dot(Xi,c.T)+error 	  

	#Test Set
	t=np.random.uniform(-1,1,[100,2])
	ti=np.insert(t,0,1,axis=1)
	tY=np.dot(ti,c.T)+error

	#Train model
	for i in xrange(200):
		print "Loss Function : ",nn_model.train(X,Y)

	#Predictions
	print "Predicted values :"
	print nn_model.predict(t)	


elif testcase=='2':
	#L2 loss regression test

	#Configuration
	opti=Optimizer(0.1,2)
	nn_model = DenseNet(2,opti,"L1")
	nn_model.addlayer('Linear',1)
	
	#Model Training set: y=c0+5*x1+3*x2+error
	c=np.array([[2,5,3]])
	error=np.random.uniform(-1,1,[100,1])		
	X=np.random.uniform(-1,1,[100,2])
	Xi=np.insert(X,0,1,axis=1)
	Y=np.dot(Xi,c.T)+error

	#Test Set
	t=np.random.uniform(-1,1,[100,2])
	ti=np.insert(t,0,1,axis=1)
	tY=np.dot(ti,c.T)+error

	#Train model
	for i in xrange(100):
		print "Loss Function : ",nn_model.train(X,Y)
	print "Predicted values :"
	print nn_model.predict(t)	



elif testcase=='3':
	#Softmax classification test

	#Configuration
	opti=Optimizer(0.1,4)
	nn_model = DenseNet(2,opti,"Softmax")
	nn_model.addlayer('ReLU',4)
	nn_model.addlayer('Linear',2)
	nn_model.addlayer('Softmax',2)

	#Model Training set: XOR
	x = np.array([  [0,1],[1,0],[1,1],[0,0]  ])
	y = np.array([  [1,0],[1,0],[0,1],[0,1]  ])

	#Test Set
	xt=np.array([  [0,0],[1,1],[0,1],[1,0] ])

	#Train model
	for i in xrange(500):
		print "Loss Function : ",nn_model.train(x,y)

	#Predictions
	print "Predicted values :"
	print nn_model.predict(xt)


elif testcase=='4':
	#SVM loss classification test

	#Configuration
	opti=Optimizer(0.1,0.6)
	nn_model = DenseNet(2,opti,"SVM")
	nn_model.addlayer('Linear',3)

	#Model Training set
	x=np.array([[0.50,0.40],[0.80,0.30],[0.30,0.80],[-0.40,0.30],[-0.30,0.70],[-0.70,0.20],[0.70,-0.40],[0.50,-0.60],[-0.40,-0.50]])
	y=np.array([[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1]])

	#Test Set
	xt=np.array([[0.50,0.40],[-0.40,0.30],[-0.30,0.70],[-0.70,0.20],[0.70,-0.40],[0.50,-0.60],[-0.40,-0.50],[0.80,0.30],[0.30,0.80]])

	#Train model
	for i in xrange(500):
		print "Loss Function : ",nn_model.train(x,y)

	#Predictions
	print "Predicted values : "
	print nn_model.predict(xt) 



else: print "Only 4 test cases available"
