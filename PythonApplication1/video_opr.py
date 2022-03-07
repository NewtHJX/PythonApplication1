#from __future__ import print_function

import numpy as np
import cv2
import time

def Image(sourceDir):

	# 读取图片
	img = sourceDir
	# 灰度化
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 高斯模糊处理:去噪(效果最好)
	blur = cv2.GaussianBlur(gray, (9, 9), 0)

	# Sobel计算XY方向梯度
	gradX = cv2.Sobel(blur, ddepth=cv2.CV_32F, dx=1, dy=0)
	gradY = cv2.Sobel(blur, ddepth=cv2.CV_32F, dx=0, dy=1)
	# 计算梯度差
	gradient = cv2.subtract(gradX, gradY)
	#绝对值
	gradient = cv2.convertScaleAbs(gradient)
	#高斯模糊处理：去噪
	blured = cv2.GaussianBlur(gradient, (9, 9), 0)

	# 二值化
	_ , dst = cv2.threshold(blured, 30, 255, cv2.THRESH_BINARY) #90为分界
	# 滑动窗口
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (107, 76))
	# 形态学处理:形态闭处理(腐蚀)
	closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
	# 腐蚀与膨胀迭代
	closed = cv2.erode(closed, None, iterations=3)
	closed = cv2.dilate(closed, None, iterations=3)
	cv2.imshow("Box", closed)
	# 获取轮廓,
	cnts,he= cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	#print(len(cnts))
	try:
		c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
	except:
		return
	#框出最小长方形
	rect = cv2.minAreaRect(c)
	#print(rect[0])#center of rect
	cmd = displacement(rect)
	#print(cmd)
	box = np.int0(cv2.boxPoints(rect))

	draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
	cv2.imshow("Box", draw_img)

def cap_video():
    cap = cv2.VideoCapture(0) #接通后0为usb摄像头
    if cap.isOpened():
        print("camera is on\n")
        show_video(cap)
    else:
        print("fail to open camera\n")

def show_video(cap):
    while 1:
        ret,frame = cap.read()
        if ret:
            time.clock()
            Image(frame)
            #cv2.waitKey(60)
            if cv2.waitKey(20) & 0xff == ord('q'): break
        else :
            print("failed\n")


def displacement(rect):
	# put the data in a list
    pub = []
    cmd = []
    for item in rect:
        if isinstance(item, tuple):
            for x in item:
                pub.append(x)
        else:
            pub.append(item)
    global pub_pre
    if len(pub_pre):
        cmd = [a - b for a, b in zip(pub, pub_pre)]        
    pub_pre = pub
    return cmd
       

pub_pre = []
cap_video()
