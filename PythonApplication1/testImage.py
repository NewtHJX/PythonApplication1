import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread("image2.jpg",0)

img = cv2.GaussianBlur(img, (9, 9), 0)
 
edges = cv2.Canny(img,50,180,True)
 
plt.subplot(121),plt.imshow(img,cmap ="gray")
plt.title("Orignal"),plt.xticks([]),plt.yticks([])
print(type(img))

plt.subplot(122),plt.imshow(edges,cmap="gray")
plt.title("Edge Image"),plt.xticks([]),plt.yticks([])


plt.show()
