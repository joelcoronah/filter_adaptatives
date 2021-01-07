import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/5.tif', 0)

# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
width = 400
height = 400
dim = (width, height)

resized1 = cv2.resize(th3, dim, interpolation=cv2.INTER_AREA)
resized2 = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
resized3 = cv2.resize(th2, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('thresh otsu', resized1)
cv2.imshow('thresh', resized3)
cv2.imshow('image', resized2)
cv2.waitKey()


# for i in xrange(3):
#     plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()
