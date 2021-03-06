import numpy as np
import cv2

width = 400
height = 400
dim = (width, height)

gray = cv2.imread('img/4.tif', cv2.IMREAD_GRAYSCALE)

resized1 = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('tutorial umbral', resized1)

# umbral fijo
_, dst1 = cv2.threshold(gray, 96, 255, cv2.THRESH_BINARY)

resized2 = cv2.resize(dst1, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('umbral fijo', resized2)

# umbral adaptable
gray = cv2.medianBlur(gray, 5)
dst2 = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55, 43)

resized3 = cv2.resize(dst2, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('umbral adaptable', resized3)
cv2.waitKey()

# cv2.waitKey(0)
