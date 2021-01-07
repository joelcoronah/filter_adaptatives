import cv2 as cv
import numpy as np

#Load and normalize .tif image:
img = 'img/4.tif'
img = cv.imread(img, 0)
img_scaled = cv.normalize(img,  np.zeros((800, 800)), 0, 255, cv.NORM_MINMAX)

#Resizing images:
pct = 30
w = int(img.shape[1] * pct / 100)
h = int(img.shape[0] * pct / 100)
dim = (w, h)
resized = cv.resize(img_scaled, dim, interpolation = cv.INTER_AREA)

# Etapa 1: Filtro Gaussiano.
gaussian = cv.GaussianBlur(resized,(5, 5),0)
# Etapa 2: detección de bordes.
low_threshold = 90
high_threshold = 150
edges = cv.Canny(gaussian, low_threshold, high_threshold)
# Etapa 3: Transformada de Hough Line.
rho = 1
theta = np.pi / 180
threshold = 15
minLineLenght = 15
maxLineGap = 15
myBlankLines = resized.copy()
lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    minLineLenght, maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv.line(myBlankLines,(x1,y1),(x2,y2),(255,0,0),5)


cv.imshow('image_resized', resized)
cv.imshow('Stage 1: Gaussian', gaussian)
cv.imshow('Stage 2: Edges', edges)
cv.imshow('Stage 3: Lines', myBlankLines)
cv.waitKey(0)
cv.destroyAllWindows()