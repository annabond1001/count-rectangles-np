import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
	return x * (1 - x)

class NeuralNetwork:
	def __init__(self, input_size, output_size):
		
		self.weights1 = np.zeros([input_size, 1000])
		self.weights2 = np.zeros([1000, 300])
		self.weights3 = np.zeros([300, 100])
		self.weights4 = np.zeros([100, output_size])

		self.bias1	= np.zeros([1, 1000])
		self.bias2	= np.zeros([1, 300])
		self.bias3	= np.zeros([1, 100])
		self.bias4	= np.zeros([1, output_size])

		self.lr		= 0.01
		print "Initializing neural network with {} parameters".format(self.weights1.size + self.weights2.size + self.weights3.size + self.weights4.size)

	def set_learning_rate(self, lr):
		self.lr		= lr

	def forward(self, x, y):
		self.input  = np.reshape(x, (1, x.size))
		self.y		= np.reshape(y, (1, y.size))
		self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.bias1)
		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2)+ self.bias2)
		self.layer3 = sigmoid(np.dot(self.layer2, self.weights3)+ self.bias3)
		self.output = np.dot(self.layer3, self.weights4)		+ self.bias4
	
	def backprop(self):
		dloss_o		= 2 * (self.y - self.output)
		dloss_l3	= np.dot(dloss_o, self.weights4.T) * d_sigmoid(self.layer3)
		dloss_l2	= np.dot(dloss_l3, self.weights3.T) * d_sigmoid(self.layer2)
		dloss_l1	= np.dot(dloss_l2, self.weights2.T) * d_sigmoid(self.layer1)

		d_weights4  = np.dot(self.layer3.T, dloss_o)
		d_weights3  = np.dot(self.layer2.T, dloss_l3)
		d_weights2  = np.dot(self.layer1.T, dloss_l2)
		d_weights1	= np.dot(self.input.T, dloss_l1)

		self.weights1 += self.lr * d_weights1
		self.weights2 += self.lr * d_weights2
		self.weights3 += self.lr * d_weights3
		self.weights4 += self.lr * d_weights4

		d_bias4		= dloss_o
		d_bias3		= dloss_l3
		d_bias2		= dloss_l2
		d_bias1		= dloss_l1

		self.bias1	+= self.lr * d_bias1
		self.bias2	+= self.lr * d_bias2
		self.bias3	+= self.lr * d_bias3
		self.bias4	+= self.lr * d_bias4