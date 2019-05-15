#Toy example for predicting the number of ones in a given binary array
#eg: [0,1,1,0,0,1,0] -> 3

import numpy as np
from matplotlib import pyplot as plt
from random import randint, seed

from net import NeuralNetwork
seed(3)

N = 20
nn = NeuralNetwork(N, 1)
n_iter = 10000
losses = []
for i in range(n_iter):
	y = np.array([randint(0, N)])
	x = np.concatenate([np.ones([y[0]]), np.zeros([N-y[0]])], axis=0)
	np.random.shuffle(x)
	nn.forward(x, y)
	nn.backprop()
	loss = pow(y-nn.output, 2)[0][0]
	losses.append(loss)
	print i,loss, "acutal: {} | predict: {}".format(y[0], nn.output[0])

y = np.array([5])
x = np.concatenate( [np.ones([1, y[0]]), np.zeros([1,  N-y[0]])], axis=1)
nn.forward(x, y)
print "Actual output: {} | Predicted output: {}".format(y, nn.output)

plt.plot(range(len(losses)), losses)
plt.show()
