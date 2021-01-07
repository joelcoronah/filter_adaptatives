import cv2 as cv
import numpy as np

# Load and normalize .tif image:
img = 'img/prueba.tif'
img = cv.imread(img)
# img_scaled = cv.normalize(img,  np.zeros((800, 800)), 0, 255, cv.NORM_MINMAX)

# Resizing images:
pct = 30
w = int(img.shape[1] * pct / 100)
h = int(img.shape[0] * pct / 100)
dim = (w, h)
image = cv.resize(img, dim, interpolation=cv.INTER_AREA)

blur = cv.medianBlur(image, 7)
gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

# umbral adaptable
thresh = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)

# thresh = cv.threshold(gray, 160, 255, cv.THRESH_BINARY_INV)[1]

cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    cv.drawContours(image, [c], -1, (36, 255, 12), 2)

cv.imshow('thresh', thresh)
cv.imshow('image', image)
cv.waitKey()
