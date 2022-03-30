# coding=utf-8
# -*- coding: utf-8 -*

import numpy as np

import cv2
from matplotlib import pyplot as plt
def pause():
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def removeBackground(color_image,depth_image,clipping_distance):
    #remove background - 将clips_distance以外的像素设置为灰色
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    
    return bg_removed

sourceDir = "0.png"
def Image(sourceDir):

	# 读取图片
	img = cv2.imread(sourceDir)

	depth_image = np.load("./0.npy")
	img_no_change=img

	plt.subplot(111),plt.imshow(depth_image)
	plt.title("Depth"),plt.xticks([]),plt.yticks([])
	plt.show()

	cv2.imshow('img',img)
	pause()

	mask=minus_hand(img) #除去手后的图像
	cv2.imshow("mask",mask)
	pause()
	#dst = cv2.bitwise_not(dst,mask)
	mask = np.dstack((mask,mask,mask))
	img = np.where((mask == 255), 153, img)
	
	cv2.imshow("Dst2", img)#二值化后图像
	pause()

	depth_scale=0.0002500000118743628
	clipping_distance_in_meters = 0.7 #1 meter
	clipping_distance = clipping_distance_in_meters / depth_scale
	img=removeBackground(img,depth_image,clipping_distance)

	cv2.imshow("Dst2", img)#二值化后图像
	pause()

	# 灰度化
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	cv2.imshow('gray',gray)
	pause()

	# 高斯模糊处理:去噪(效果最好)
	blur = cv2.GaussianBlur(gray, (9, 9), 0)

	# Sobel计算XY方向梯度
	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)

	#cv2.imshow('gradient',gradX)
	#pause()

	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)

	#cv2.imshow('gradient',gradY)
	#pause()

	#gradient = cv2.subtract(gradX,gradY) #计算梯度差
		
	# 计算梯度差
	gradient = cv2.subtract(gradX, gradY)

	#绝对值
	gradient = cv2.convertScaleAbs(gradient)

	cv2.imshow('gradient',gradient)
	pause()

	#高斯模糊处理：去噪
	blured = cv2.GaussianBlur(gradient, (9, 9), 0)
	
	
	# 二值化
	_ , dst = cv2.threshold(blured, 20, 255, cv2.THRESH_BINARY) #90为分界

	cv2.imshow('dst',dst)
	pause()

	#mask=minus_hand(img) #除去手后的图像
	#cv2.imshow("mask",mask)
	#pause()
	##dst = cv2.bitwise_not(dst,mask)

	#dst = np.where((mask == 255), 0, dst)

	#cv2.imshow("Dst2", dst)#二值化后图像
	#pause()

	# 滑动窗口
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45,67));#print(kernel) #1280,960
	
	# 形态学处理:形态闭处理(腐蚀)
	closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

	#cv2.imshow('closed1',closed)
	#pause()

	# 腐蚀与膨胀迭代
	closed = cv2.erode(closed, None, iterations=2)
	closed = cv2.dilate(closed, None, iterations=3)

	cv2.imshow('closed2',closed)
	pause()

	# 获取轮廓,
	cnts,he= cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	#cv2.imshow('cnts1',cnts)
	#print(cnts)
	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
	#print("ccccccccccccccc:\n",c)
	#c = sorted(closed, key=cv2.contourArea, reverse=True)[0]
	
	##rect = cv2.minAreaRect(c)
	rect = cv2.minAreaRect(c)

	#cv2.imshow('rect',rect)
	#print(rect)

	box = np.int0(cv2.boxPoints(rect))


	draw_img = cv2.drawContours(img_no_change.copy(), [c], -1, (0, 0, 255), 3)
	cv2.imshow("Box", draw_img)
	#cv2.imwrite('image_test6.png', draw_img)
	pause()

	draw_img = cv2.drawContours(img_no_change.copy(), [box], -1, (0, 0, 255), 3)
	cv2.imshow("Box", draw_img)
	#cv2.imwrite('image_test6.png', draw_img)
	pause()
	#cv2.waitKey(0)


def minus_hand(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	low_hand = np.array([9, 40, 0])#np.array([9, 40, 0])
	high_hand = np.array([60, 255, 255])#np.array([43, 255, 255])
	mask = cv2.inRange(hsv, low_hand, high_hand)
	plt.subplot(221),plt.imshow(hsv)
	plt.title("Depth"),plt.xticks([]),plt.yticks([])

	mask = cv2.dilate(mask, None, iterations=6)

	mask=cv2.multiply(mask,255)
	plt.subplot(222),plt.imshow(mask)
	plt.title("Depth"),plt.xticks([]),plt.yticks([])
	plt.show()

	return mask


Image(sourceDir)
#print("image:")
#sourceDir = "image.jpg"
#img2 = cv2.imread('image.jpg',1)
#cv2.imshow('1',img2)

cv2.waitKey(0)
cv2.destroyAllWindows()



