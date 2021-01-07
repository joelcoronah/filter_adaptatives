import cv2 as cv
import numpy as np

# Load and normalize .tif image:
img = 'img/4.tif'
img = cv.imread(img)
img_scaled = cv.normalize(img,  np.zeros((800, 800)), 0, 255, cv.NORM_MINMAX)

# Resizing images:
pct = 30
w = int(img.shape[1] * pct / 100)
h = int(img.shape[0] * pct / 100)
dim = (w, h)
resized = cv.resize(img_scaled, dim, interpolation=cv.INTER_AREA)
cv.imshow('image_resized', resized)

# create 1 pixel red image
red = np.zeros((1, 1, 3), np.uint8)
red[:] = (0,0,255)

# create 1 pixel blue image
blue = np.zeros((1, 1, 3), np.uint8)
blue[:] = (255,0,0)

# append the two images
lut = np.concatenate((red, blue), axis=0)

# resize lut to 256 values
lut = cv.resize(lut, (1,256), interpolation=cv.INTER_CUBIC)

# apply lut
resized = cv.LUT(resized, lut)

# First stage: Gaussian blur.
gaussian = cv.GaussianBlur(resized, (5, 5), 0)

# Second Stage: edge detection with Canny.
low_threshold = 90
high_threshold = 150
edges = cv.Canny(gaussian, low_threshold, high_threshold)

# plus, with 2d filter
kernel = np.ones((5, 5), np.float32)/25
dst = cv.filter2D(resized, -1, kernel)

# edge detection with Canny.
low_threshold = 90
high_threshold = 150
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


cv.imshow('Stage 1: Dst', dst)
cv.imshow('Stage 2: Edges', edges)
cv.imshow('Stage 2: Edges dst', edges_dst)

cv.imshow('RGB', resized)

# cv.imshow('color RBG', after)

cv.waitKey(0)
cv.destroyAllWindows()
