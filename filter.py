import cv2 as cv
import numpy as np

# Load and normalize .tif image:
img = 'img/prueba.tif'
img = cv.imread(img, 0)
img_scaled = cv.normalize(img,  np.zeros((800, 800)), 0, 255, cv.NORM_MINMAX)

# Resizing images:
pct = 30
w = int(img.shape[1] * pct / 100)
h = int(img.shape[0] * pct / 100)
dim = (w, h)
resized = cv.resize(img_scaled, dim, interpolation=cv.INTER_AREA)

# First stage: Gaussian blur.
gaussian = cv.GaussianBlur(resized, (5, 5), 0)

# Second Stage: edge detection with Canny.
low_threshold = 250
high_threshold = 255
edges = cv.Canny(gaussian, low_threshold, high_threshold)

# plus, with 2d filter
kernel = np.ones((5, 5), np.float32)/25
dst = cv.filter2D(resized, -1, kernel)

# edge detection with Canny.
low_threshold = 250
high_threshold = 255
edges_dst = cv.Canny(dst, low_threshold, high_threshold)


# Third stage: Hough Line transform.
rho = 1
theta = np.pi / 180
threshold = 15
minLineLenght = 15
maxLineGap = 15
myBlankLines = resized.copy()
lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                       minLineLenght, maxLineGap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv.line(myBlankLines, (x1, y1), (x2, y2), (255, 0, 0), 5)


cv.imshow('image_resized', resized)
cv.imshow('Stage 1: Dst', dst)
cv.imshow('Stage 2: Edges', edges)
cv.imshow('Stage 2: Edges dst', edges_dst)
# cv.imshow('Stage 3: Lines', myBlankLines)
cv.waitKey(0)
cv.destroyAllWindows()
