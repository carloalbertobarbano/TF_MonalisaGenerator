from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
import os
import sys


fig_1 = plot.figure(1)
fig_2 = plot.figure(2)
fig_3 = plot.figure(3)

plot_red = fig_1.add_subplot(111, projection='3d')
plot_green = fig_2.add_subplot(111, projection='3d')
plot_blue = fig_3.add_subplot(111, projection='3d')

plot_x = []
plot_y = []
plot_r = []
plot_g = []
plot_b = []

X = []
Y = []

img_name = "monalisa"
monalisa = Image.open("{0}.jpg".format(img_name))
w, h = monalisa.size
pix = monalisa.load()

num = 0
for i in range(w):
	for j in range(h):
		X.append([i / w, j / h])
		r, g, b = pix[i, j]
		Y.append([r / 255, g / 255, b / 255])
		if num % 100 == 0:
			plot_x.append(i)
			plot_y.append(j)
			plot_r.append(r)
			plot_g.append(g)
			plot_b.append(b)
		num = num + 1

plot_red.scatter(plot_x, plot_y, plot_r, c='r', marker='o')
plot_red.set_xlabel('X')
plot_red.set_ylabel('Y')
plot_red.set_zlabel('Red value')


plot_green.scatter(plot_x, plot_y, plot_g, c='g', marker='o')
plot_green.set_xlabel('X')
plot_green.set_ylabel('Y')
plot_green.set_zlabel('Green value')


plot_blue.scatter(plot_x, plot_y, plot_b, c='b', marker='o')
plot_blue.set_xlabel('X')
plot_blue.set_ylabel('Y')
plot_blue.set_zlabel('Blue value')
plot.show()