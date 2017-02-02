from PIL import Image
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import os
import sys

tf.logging.set_verbosity(tf.logging.ERROR)

X = []
Y = []

img_name = "monalisa"
monalisa = Image.open("{0}.jpg".format(img_name))
w, h = monalisa.size
pix = monalisa.load()

for i in range(w):
	for j in range(h):
		X.append([i / w, j / h])
		r, g, b = pix[i, j]
		Y.append([r / 255, g / 255, b / 255])

X_test = X
Y_test = Y

epochs = []
error = []

print("Data size: ")
print("X: ", len(X))
print("Y: ", len(Y))
print("X_test: ", len(X_test))
print("Y_test: ", len(Y_test))


tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
network = input_data(shape=[None, 2])
network = fully_connected(network, 350, activation='sigmoid', weights_init=tnorm)
network = fully_connected(network, 350, activation='sigmoid', weights_init=tnorm)
network = fully_connected(network, 350, activation='sigmoid', weights_init=tnorm)
network = fully_connected(network, 350, activation='sigmoid', weights_init=tnorm)
network = fully_connected(network, 350, activation='sigmoid', weights_init=tnorm)
network = fully_connected(network, 3, activation='sigmoid', weights_init=tnorm)

learning_rate = 0.8
regr = regression(network, optimizer='sgd', loss='mean_square',
					learning_rate = learning_rate)
model = tflearn.DNN(regr)

if len(sys.argv) > 1:
	model.load(sys.argv[1])
	print("Model loaded from: " + sys.argv[1])

for n in range(1000):
	model.fit(X, Y, n_epoch=10, snapshot_epoch=False)
	if len(sys.argv) > 1:
		model.save(sys.argv[1])
		print("Model saved to " + sys.argv[1])

	print("Model trained. Trying to predict")

	monalisa_output = Image.new("RGB", (w, h), "white")
	pix_output = monalisa_output.load()

	p = model.predict(X)

	pp = 0
	for i in range(w):
		for j in range(h):
			r = int(p[pp][0] * 255)
			g = int(p[pp][1] * 255)
			b = int(p[pp][2] * 255)
			pix_output[i, j] = (r, g, b)

			pp = pp + 1

	monalisa_output.save(img_name + "_output.jpg")
	
	
	padding = ""  #ls -l output will be sorted, yay
	if (n+1)*10 < 1000:
		padding = "0"

	if (n + 1)*10 < 100:
		padding = "00"

	outpath = "outputs/6_layers_350_{0}/".format(learning_rate)
	os.system("cp " + img_name + "_output.jpg " + outpath + img_name + "_output_{0}{1}.jpg".format(padding, (n+1)*10))
	print("Image saved to {0}_output.jpg".format(img_name))
	print("Image copied to " + outpath + img_name + "_output_{0}{1}.jpg".format(padding, (n+1)*10))

