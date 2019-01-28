import cv2
from matplotlib import pyplot as plt
import numpy as np

im = cv2.imread('C:/Users/PelaoT/Desktop/Practica/dataset/normalizado/testHeStain/A02_03.bmp')
im_copy = np.copy(im)
im2 = np.copy(im)
b, g, r = cv2.split(im_copy)
b = b.astype(np.float64)
g = g.astype(np.float64)
r = r.astype(np.float64)
br = (100 * b / (1 + r + g)) * (256 / (1 + b + r + g))
br = br.astype(np.uint8)

mean, std = cv2.meanStdDev(br)
mean = mean[0][0]
std = std[0][0]

kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(br, cv2.MORPH_OPEN, kernel)

hist, bins = np.histogram(thresh.ravel(), 256, [0,256])
umbral = mean + std
max_hist = np.max(hist)
plt.plot(hist)
plt.plot([umbral, umbral], [0, max_hist])
plt.text(int(umbral) + 1, max_hist - (max_hist * 0.1), int(umbral))
plt.tick_params(axis='y',
                which='both',
                right='off',
                left='off',
                labelleft='off')
plt.show()
