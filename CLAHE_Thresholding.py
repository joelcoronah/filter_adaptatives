from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/4.tif', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(img)

plt.hist(cl_img.flat, bins=100, range=(0, 255))

ret, th1 = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


img_rs = cv2.resize(img, (800,600))
img_th1 = cv2.resize(th1, (800,600))


cv2.imshow('origin img', img_rs)
cv2.imshow('otsu img', img_th1)


cv2.waitKey(0)
cv2.destroyAllWindows()