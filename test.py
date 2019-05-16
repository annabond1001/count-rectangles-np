import numpy as np
from matplotlib import pyplot as plt
from random import randint, seed
import pickle

from net import NeuralNetwork
seed(29)

SIZE = 32
MAX_N_RECT = 10
MIN_RECT_SIZE = 6
def generate_sample(n_rect):
	canvas	= np.zeros([SIZE, SIZE])
	for i in range(n_rect):
		rect_found = False
		while(not rect_found):
			left	= randint(0, SIZE)
			top		= randint(0, SIZE)
			right	= randint(0, SIZE)
			bottom	= randint(0, SIZE)
			if(right-left >= MIN_RECT_SIZE and bottom-top >= MIN_RECT_SIZE):
				rect_found = True
				for x in range(left, right):
					for y in range(top, bottom):
						canvas[y][x] += 1 #canvas[y][x] = 1 #for combined boxes
	return (canvas, n_rect)


canvas, y = generate_sample(1)
#nn = NeuralNetwork(canvas.size, 1)
#loading model
with open("models/model.pkl") as f:
	nn = pickle.load(f)
n_iter = 10
losses = []
for i in range(n_iter):
	n_rect	= randint(0, MAX_N_RECT)
	canvas, y = generate_sample(n_rect)
	nn.forward(np.concatenate(canvas), np.array([y]))
	plt.imshow(canvas, interpolation="nearest")
	plt.suptitle("Actual output: {} | Predicted output: {}".format(y, nn.output[0]))
	plt.show()
