import cv2
import numpy as np
img = "image3.jpg"
img =cv2.imread(img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
np.set_printoptions(threshold=np.inf)
print(np.array(hsv))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#dst = cv2.bitwise_xor(gray,mask)
#cv2.imshow("test",dst)