import cv2
import numpy as np

sourceDir = "image.jpg"
img2 = cv2.imread(sourceDir,1)
cv2.imshow('test1',img2)

img = cv2.imread('image.jpg',1)

cv2.imshow('imshow',img)

cv2.waitKey(0)

cv2.destroyAllWindows()