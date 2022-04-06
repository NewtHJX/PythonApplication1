import cv2
import numpy as np
from matplotlib import pyplot as plt

def canny(img,imgd):

    blur = cv2.GaussianBlur(img, (9, 9), 0)
 
    edges = cv2.Canny(blur,0,60)
 
    plt.subplot(121),plt.imshow(img,cmap ="gray")
    plt.title("Orignal"),plt.xticks([]),plt.yticks([])

    plt.subplot(122),plt.imshow(edges,cmap="gray")
    plt.title("Edge Image"),plt.xticks([]),plt.yticks([])

    plt.show()
    return edges

def minus_hand(img):
	#RGB转换到HSV空间
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#手的颜色范围
	low_hand = np.array([9, 20, 0])#np.array([9, 40, 0])
	high_hand = np.array([70, 255, 255])#np.array([43, 255, 255])
	#生成掩膜
	mask = cv2.inRange(hsv, low_hand, high_hand)
	#plt.subplot(221),plt.imshow(hsv)
	#plt.title("Depth"),plt.xticks([]),plt.yticks([])
	#plt.subplot(222),plt.imshow(mask)
	#plt.title("Depth"),plt.xticks([]),plt.yticks([])
	#plt.show()

	return mask

def pause():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cutImage(color_image, depth_image):

	#plt.subplot(111),plt.imshow(depth_image)
	#plt.title("Depth"),plt.xticks([]),plt.yticks([])
	#plt.show()

	#cv2.imshow("depth_image",depth_image)
	#pause()
	
	##返回手的掩膜
	#mask = minus_hand( color_image )
	
	##cv2.imshow("mask",cv2.multiply(mask,255))
	##pause()

	##中值滤波#效果很好啊
	#img_medianBlur=cv2.medianBlur(mask,5)
	##cv2.imshow("mask",cv2.multiply(img_medianBlur,255))
	##pause()

	##对掩膜进行膨胀
	#mask_dilate = cv2.dilate(img_medianBlur, None, iterations=6)
	##掩膜变为白色 可视化
	#mask_show=cv2.multiply(mask_dilate,255)
	#cv2.imshow("mask_dilate",mask_show)
	##pause()
	##dst = cv2.bitwise_not(dst,mask)

	#mask_dilate_3d = np.dstack((mask_dilate,mask_dilate,mask_dilate))
	#color_image_minus_hand = np.where((mask_dilate_3d == 255), 153, color_image)
	
	#cv2.imshow("Color_image_minus_hand", color_image_minus_hand)#二值化后图像
	#pause()

	##删除clipping_distance_in_meters外的背景
	#clipping_distance_in_meters = 1 #可以修改识别的距离 1 meter
	##删除背景
	#bg_removed_minus_hand = removeBackground(color_image_minus_hand,depth_image, clipping_distance_in_meters )

	#cv2.imshow("Bg_removed_minus_hand", bg_removed_minus_hand)#二值化后图像
	##pause()

	# 灰度化
	#gray = cv2.cvtColor(bg_removed_minus_hand, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

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
		
	# 计算梯度差
	gradient = cv2.subtract(gradX, gradY)

	#绝对值
	gradient = cv2.convertScaleAbs(gradient)
	

	cv2.imshow('gradient',gradient)
	pause()

	#高斯模糊处理：去噪
	blured = cv2.GaussianBlur(gradient, (9, 9), 0)
	#cv2.imwrite("blured7.png",blured)
	#cv2.imshow('blured',blured)
	#pause()	
	
	# 二值化#动态二值化？？？？？？？？
	
	#res1 = cv2.adaptiveThreshold(blured,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,3)
	#cv2.imshow('res1',res1)
	#pause()
	_ , dst = cv2.threshold(blured, 30, 255, cv2.THRESH_BINARY) #90为分界

	cv2.imshow('dst',dst)
	pause()

	# 滑动窗口#这里的数值可以改
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45,67));#print(kernel) #1280,960
	
	# 形态学处理:形态闭处理(腐蚀)
	closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

	cv2.imshow('closed1',closed)
	pause()

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

if __name__ == "__main__" :
	plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

	color_image = cv2.imread("./7.png")
	depth_image = np.load("./7.npy")
	bg_removed = cv2.imread("bg_removed7.png")
	hand_removed = cv2.imread("hand_removed7.png")
	blured = cv2.imread("blured7.png")

	cutImage( color_image, depth_image)
	cv2.waitKey(0)

	##canny( color_image, depth_image )
	#edg = canny( bg_removed, depth_image )
	#mask = minus_hand( bg_removed )

	##中值滤波#效果很好啊
	#img_medianBlur=cv2.medianBlur(mask,5)
	#cv2.imshow("mask",cv2.multiply(img_medianBlur,255))
	##pause()

	##对掩膜进行膨胀
	#mask_dilate = cv2.dilate(img_medianBlur, None, iterations=4)
	##掩膜变为白色 可视化
	#mask_dilate = mask_dilate
	#mask_show = cv2.multiply(mask_dilate,255)
	#cv2.imshow("mask_dilate",mask_show)
	##pause()
	##dst = cv2.bitwise_not(dst,mask)

	#mask_dilate_3d = np.dstack((mask_dilate,mask_dilate,mask_dilate))
	#color_image_minus_hand = np.where((mask_dilate_3d == 255), 0, blured)
	#cv2.imshow("color_image_minus_hand",color_image_minus_hand)
	#cv2.waitKey(0)



	#canny( hand_removed, depth_image )
    
