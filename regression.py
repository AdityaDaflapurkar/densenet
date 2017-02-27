import numpy as np

class Linear_Regression:
	def __init__(self):	
		self.coefficients=np.random.rand(1,2)	
	
	def train(self,X,Y,iterations):
		for epoch in xrange(iterations):
			for row in xrange(len(X))
				predicted=self.predict(X[row])
				error=Y[row]-predicted
				self.coefficients=self.coefficients-0.1*error*X[row]
				
	def predict(self,X):
		result=np.dot(np.transpose(self.coefficients),X)
		return result

	if __name__=='__main__':
		model=Linear_Regression()
		dataset = list()
		f=open('winequality-white.csv', 'r')
		csv_reader = reader(f)
		for row in csv_reader:
			row=[1]+row
			if not row:
				continue
			dataset.append(row)
		for i in range(len(dataset[0])):
			str_column_to_float(dataset, i)
		model.train()
