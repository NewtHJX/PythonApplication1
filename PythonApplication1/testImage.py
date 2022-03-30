import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread("7.png",0)

blur = cv2.GaussianBlur(img, (9, 9), 0)
 
edges = cv2.Canny(blur,0,90)
 
plt.subplot(121),plt.imshow(img,cmap ="gray")
plt.title("Orignal"),plt.xticks([]),plt.yticks([])

plt.subplot(122),plt.imshow(edges,cmap="gray")
plt.title("Edge Image"),plt.xticks([]),plt.yticks([])


plt.show()
