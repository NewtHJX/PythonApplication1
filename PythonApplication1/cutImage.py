# coding=utf-8
# -*- coding: utf-8 -*

import numpy as np
import cv2
from matplotlib import pyplot as plt

def pause():
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def removeBackground(color_image,depth_image, clipping_distance_in_meters):
	# 我们将删除对象的背景
	#  clipping_distance_in_meters meters away
	depth_scale=0.0002500000118743628
	clipping_distance = clipping_distance_in_meters / depth_scale
	#remove background - 将clips_distance以外的像素设置为灰色
	grey_color = 153
	depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
	bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    
	return bg_removed

def cutImage(color_image, depth_image):
	#读取图片
	#彩色图片color_image 深度图片depth_image
	#img = color_image
	#img_no_change=img

	#plt.subplot(111),plt.imshow(depth_image)
	#plt.title("Depth"),plt.xticks([]),plt.yticks([])
	#plt.show()

	#cv2.imshow("depth_image",depth_image)
	#pause()
	
	#返回手的掩膜
	mask = minus_hand( color_image )
	
	#cv2.imshow("mask",cv2.multiply(mask,255))
	#pause()

	#中值滤波#效果很好啊
	img_medianBlur=cv2.medianBlur(mask,5)
	#cv2.imshow("mask",cv2.multiply(img_medianBlur,255))
	#pause()

	#对掩膜进行膨胀
	mask_dilate = cv2.dilate(img_medianBlur, None, iterations=6)
	#掩膜变为白色 可视化
	mask_show=cv2.multiply(mask_dilate,255)
	cv2.imshow("mask_dilate",mask_show)
	#pause()
	#dst = cv2.bitwise_not(dst,mask)

	mask_dilate_3d = np.dstack((mask_dilate,mask_dilate,mask_dilate))
	color_image_minus_hand = np.where((mask_dilate_3d == 255), 153, color_image)
	
	#cv2.imshow("Color_image_minus_hand", color_image_minus_hand)#二值化后图像
	#pause()

	#删除clipping_distance_in_meters外的背景
	clipping_distance_in_meters = 1 #可以修改识别的距离 1 meter
	#删除背景
	bg_removed_minus_hand = removeBackground(color_image_minus_hand,depth_image, clipping_distance_in_meters )

	cv2.imshow("Bg_removed_minus_hand", bg_removed_minus_hand)#二值化后图像
	#pause()

	# 灰度化
	gray = cv2.cvtColor(bg_removed_minus_hand, cv2.COLOR_BGR2GRAY)

	#cv2.imshow('gray',gray)
	#pause()

	# 高斯模糊处理:去噪(效果最好)#这里没用模糊后的图像
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

	#cv2.imshow('gradient',gradient)
	#pause()

	#高斯模糊处理：去噪
	blured = cv2.GaussianBlur(gradient, (9, 9), 0)
	
	
	# 二值化
	_ , dst = cv2.threshold(blured, 20, 255, cv2.THRESH_BINARY) #90为分界

	#cv2.imshow('dst',dst)
	#pause()

	# 滑动窗口#这里的数值可以改
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45,67));#print(kernel) #1280,960
	
	# 形态学处理:形态闭处理(腐蚀)
	closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

	#cv2.imshow('closed1',closed)
	#pause()

	# 腐蚀与膨胀迭代
	closed = cv2.erode(closed, None, iterations=3)
	closed = cv2.dilate(closed, None, iterations=3)

	#cv2.imshow('closed2',closed)
	#pause()

	# 获取轮廓,
	cnts,he= cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	#cv2.imshow('cnts1',cnts)
	#print(cnts)
	try:
		c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
	except:
		return
	#print("ccccccccccccccc:\n",c)
	#c = sorted(closed, key=cv2.contourArea, reverse=True)[0]
	
	##rect = cv2.minAreaRect(c)
	rect = cv2.minAreaRect(c)

	#cv2.imshow('rect',rect)
	#print(rect)

	box = np.int0(cv2.boxPoints(rect))


	draw_img_outline = cv2.drawContours(color_image.copy(), [c], -1, (0, 0, 255), 3)
	cv2.imshow("Outline", draw_img_outline)
	#cv2.imwrite('image_test6.png', draw_img)
	#pause()

	#draw_img_box = cv2.drawContours(color_image.copy(), [box], -1, (0, 0, 255), 3)
	#cv2.imshow("Box", draw_img_box)
	##cv2.imwrite('image_test6.png', draw_img)
	#pause()
	##cv2.waitKey(0)

def minus_hand(img):
	#RGB转换到HSV空间
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#手的颜色范围
	low_hand = np.array([9, 40, 0])#np.array([9, 40, 0])
	high_hand = np.array([50, 255, 255])#np.array([43, 255, 255])
	#生成掩膜
	mask = cv2.inRange(hsv, low_hand, high_hand)
	#plt.subplot(221),plt.imshow(hsv)
	#plt.title("Depth"),plt.xticks([]),plt.yticks([])
	#plt.subplot(222),plt.imshow(mask)
	#plt.title("Depth"),plt.xticks([]),plt.yticks([])
	#plt.show()

	return mask



if __name__ == "__main__" :
	color_image = cv2.imread("0.png")
	depth_image = np.load("./0.npy")
	cutImage( color_image, depth_image)




