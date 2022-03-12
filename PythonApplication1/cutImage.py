# coding=utf-8
# -*- coding: utf-8 -*

import numpy as np

import cv2

def pause():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def Image(sourceDir):

	# 读取图片
	img = cv2.imread(sourceDir)

	cv2.imshow('img',img)
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
	_ , dst = cv2.threshold(blured, 30, 255, cv2.THRESH_BINARY) #90为分界

	cv2.imshow('dst',dst)
	pause()

	# 滑动窗口
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200,200));print(kernel) #1280,960
	
	# 形态学处理:形态闭处理(腐蚀)
	closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

	cv2.imshow('closed1',closed)
	pause()

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


	draw_img = cv2.drawContours(img.copy(), [c], -1, (0, 0, 255), 3)
	cv2.imshow("Box", draw_img)
	cv2.imwrite('image_test.png', draw_img)
	#cv2.waitKey(0)


sourceDir = "image5.jpg"
Image(sourceDir)
#print("image:")
#sourceDir = "image.jpg"
#img2 = cv2.imread('image.jpg',1)
#cv2.imshow('1',img2)

cv2.waitKey(0)
cv2.destroyAllWindows()



