import cv2
import numpy as np
from matplotlib import pyplot as plt

def showColorImage():
    #读入一张图片
    img = cv2.imread("./7.png")
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    #显示一张图片
    #cv2.imshow("Color Image",img)
    #cv2.waitKey(0)
    ##显示3通道图片
    #_ , dst = cv2.threshold(blured, 20, 255, cv2.THRESH_BINARY)
    b,g,r = cv2.split(img)
    plt.subplot(331),plt.imshow(b)
    plt.title("Depth"),plt.xticks([]),plt.yticks([])
    plt.subplot(332),plt.imshow(g)
    plt.title("Depth"),plt.xticks([]),plt.yticks([])
    plt.subplot(333),plt.imshow(r)
    plt.title("Depth"),plt.xticks([]),plt.yticks([])
    plt.show()
    cv2.imshow("r",r)
    cv2.imshow("g",g)
    cv2.imshow("b",b)
    cv2.waitKey(0)
    #plt.figure(0)
    #plt.hist(r,bins=255)
    histr = cv2.calcHist(r,[0],None,[256],[0,255])
    histg = cv2.calcHist(g,[0],None,[256],[0,255])
    histb = cv2.calcHist(b,[0],None,[256],[0,255])
    plt.figure(1)
    plt.subplot(131),plt.plot( histr,color="r",label="R通道直方图",linestyle="-.")
    plt.legend()
    plt.subplot(132),
    plt.plot( histg,color="g",label="G通道直方图",linestyle="--")
    plt.legend()
    plt.subplot(133),
    plt.plot( histb,color="b",label="B通道直方图",linestyle="--")
    plt.legend()

    #plt.plot( hist,color="lime",label="高斯滤波直方图",linestyle="--")
    #plt.show()
    plt.figure(2)
    histr = cv2.calcHist(img,[0],None,[256],[0,255])
    histg = cv2.calcHist(img,[1],None,[256],[0,255])
    histb = cv2.calcHist(img,[2],None,[256],[0,255])
    plt.subplot(231),plt.plot( histr,color="r",label="R通道直方图",linestyle="--")
    plt.legend()
    plt.subplot(232),
    plt.plot( histg,color="g",label="G通道直方图",linestyle="--")
    plt.legend()
    plt.subplot(233),
    plt.plot( histb,color="b",label="B通道直方图",linestyle="--")
    plt.legend()

    #plt.plot( hist,color="lime",label="高斯滤波直方图",linestyle="--")
    plt.show()

def showRGBImage(color_image):
    img = color_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV )
    #red = np.zeros_like(img);
    #red[:,:] = [0,0,100];
    #img = cv2.add(img,red)
    ##img = np.where((img[2] <= 200),100 , img[2])
    #img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB )

    img2 = cv2.cvtColor( img , cv2.COLOR_RGB2GRAY)
    r,g,b = cv2.split(img)


    plt.figure(1)
    plt.subplot(151),plt.imshow(img)
    plt.subplot(155),plt.imshow(img2,"gray")
    plt.subplot(152),plt.imshow(r)
    plt.subplot(153),plt.imshow(g)
    plt.subplot(154),plt.imshow(b)
    plt.show()
    plt.close()

    plt.figure(2)
    hist1 = cv2.calcHist([img2],[0],None,[255],[1,255])
    hist2 = cv2.calcHist([r],[0],None,[255],[1,255])
    hist3 = cv2.calcHist([g],[0],None,[255],[1,255])
    hist4 = cv2.calcHist([b],[0],None,[255],[1,255])    
    
    plt.subplot(241),plt.plot( hist1,color="gray",label="灰度图像直方图",linestyle="--")
    plt.title("灰度图像直方图")
    plt.legend()
    plt.subplot(242),plt.plot( hist2,color="r",label="R通道图像直方图",linestyle="--")
    plt.title("R通道图像直方图")
    plt.legend()
    plt.subplot(243),plt.plot( hist3,color="g",label="G通道图像直方图",linestyle="--")
    plt.title("G通道图像直方图")
    plt.legend()
    plt.subplot(244),plt.plot( hist4,color="b",label="B通道图像直方图",linestyle="--")
    plt.title("B通道图像直方图")
    plt.legend()
    plt.show()
    plt.close()

    plt.figure(3)

    mask = cv2.inRange(img2,45,81)#3654
    mask = cv2.multiply(mask,255)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)
    img2 = np.where((mask != 255), 0, img2)

    mask = cv2.inRange(r,47,85)#3446
    maskr = cv2.multiply(mask,255)
    cv2.imshow("mask",maskr)
    cv2.waitKey(0)
    r = np.where((maskr != 255), 0, r)

    mask = cv2.inRange(g,45,80)#3252
    maskg = cv2.multiply(mask,255)
    cv2.imshow("mask",maskg)
    cv2.waitKey(0)
    g = np.where((maskg != 255), 0, g)

    mask = cv2.inRange(b,35,70)#3253
    maskb = cv2.multiply(mask,255)
    cv2.imshow("mask",maskb)
    cv2.waitKey(0)
    b = np.where((maskb != 255), 0, b)
    plt.subplot(341),plt.imshow(img2,"gray"),plt.title("灰色通道阈值分割结果")
    plt.subplot(342),plt.imshow(r),plt.title("R通道阈值分割结果")
    plt.subplot(343),plt.imshow(g),plt.title("G通道阈值分割结果")
    plt.subplot(344),plt.imshow(b),plt.title("B通道阈值分割结果")
    plt.show()

def showHSVImage(color_image):
    img = color_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    h,s,v = cv2.split(img)
    plt.subplot(151),plt.imshow(img)
    plt.subplot(152),plt.imshow(h)
    plt.subplot(153),plt.imshow(s)
    plt.subplot(154),plt.imshow(v)
    hist2 = cv2.calcHist([h],[0],None,[179],[1,179])#???????????
    hist3 = cv2.calcHist([s],[0],None,[255],[1,255])
    hist4 = cv2.calcHist([v],[0],None,[255],[1,255])
    plt.show()
    plt.close()
    plt.figure(2)
    #plt.subplot(241),plt.plot( hist1,color="gray",label="灰度图像直方图",linestyle="--")
    #plt.title("灰度图像直方图")
    #plt.legend()
    plt.subplot(231),plt.plot( hist2,color="r",label="H通道图像直方图",linestyle="--")
    plt.title("H通道图像直方图")
    plt.legend()
    plt.subplot(232),plt.plot( hist3,color="g",label="S通道图像直方图",linestyle="--")
    plt.title("S通道图像直方图")
    plt.legend()
    plt.subplot(233),plt.plot( hist4,color="b",label="V通道图像直方图",linestyle="--")
    plt.title("V通道图像直方图")
    plt.legend()
    plt.show()
    plt.close()

    plt.figure(3)
    mask = cv2.inRange(h,9,30)
    maskr = cv2.multiply(mask,255)
    cv2.imshow("mask",maskr)
    cv2.waitKey(0)
    h = np.where((maskr != 255), 0, h)

    mask = cv2.inRange(s,14,68)
    maskg = cv2.multiply(mask,255)
    #cv2.imshow("mask",maskg)
    #cv2.waitKey(0)
    s = np.where((maskg != 255), 0, s)

    mask = cv2.inRange(v,45,81)#3352
    maskb = cv2.multiply(mask,255)
    #cv2.imshow("mask",maskb)
    #cv2.waitKey(0)
    v = np.where((maskb != 255), 0, v)
    plt.subplot(331),plt.imshow(h),plt.title("H通道阈值分割结果")
    plt.subplot(332),plt.imshow(s),plt.title("S通道阈值分割结果")
    plt.subplot(333),plt.imshow(v),plt.title("V通道阈值分割结果")
    plt.show()

def showDepthImage(depth_image,color_image):
    #print(type(depth_image))
    #print(type(color_image))
    #cv2.imshow("depth", depth_image )
    #cv2.waitKey(0)
    #depth_image = np.asanyarray(depth_image, dtype="uint8")
    plt.figure(1)
    #depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    plt.imshow(depth_image)
    #plt.imshow( depth_image ), plt.title( "depth" )
    plt.show()
    plt.close()
    plt.figure(2)
    arr=depth_image.flatten()
    n, bins, patches = plt.hist(arr, bins=400,facecolor='green', alpha=0.75)  
    plt.show()

def removeBackground(color_image,depth_image, clipping_distance_in_meters):
	# 我们将删除对象的背景
	#  clipping_distance_in_meters meters away
	depth_scale=0.0002500000118743628
	clipping_distance = clipping_distance_in_meters / depth_scale
	#remove background - 将clips_distance以外的像素设置为灰色
	grey_color = 0
	depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
	bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    
	return bg_removed

if __name__ == "__main__" :

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    color_image = cv2.imread("./7.png")
    depth_image = np.load("./7.npy")
    clipping_distance_in_meters = 0.69
    bg_removed = removeBackground(color_image,depth_image, clipping_distance_in_meters )
    #cv2.imwrite("bg_removed0.png", bg_removed )
    #showColorImage()

    #showRGBImage( color_image )
    #showHSVImage( color_image )

    showRGBImage( bg_removed )
    #showHSVImage( bg_removed )


    #showDepthImage(depth_image,color_image)

    
    
    
    
    
    
    #img=cv2.imread("7.png")
    #img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY,img,0)
    #img2 = cv2.GaussianBlur(img1,(9,9),0)
    #hist1 = cv2.calcHist(img1,[0],None,[256],[0,255])
    ##hist2 = cv2.calcHist(img2,[0],None,[256],[0,255])
    #plt.subplot(121),plt.plot( hist1,color="r",label="灰度图像直方图",linestyle="--")#, plt.imshow(hist1)
    ##plt.title("灰度图像直方图")
    #plt.legend()
    #plt.subplot(122),
    #plt.plot( hist2,color="b",label="高斯滤波直方图",linestyle="--")
    ##plt.title("高斯滤波直方图")
    #plt.legend()

    ##plt.plot( hist,color="lime",label="高斯滤波直方图",linestyle="--")
    #plt.show()