import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__" :

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    img=cv2.imread("0.png")
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY,img,0)
    img2 = cv2.GaussianBlur(img1,(9,9),0)
    hist1 = cv2.calcHist(img1,[0],None,[256],[0,255])
    hist2 = cv2.calcHist(img2,[0],None,[256],[0,255])
    plt.subplot(121),plt.plot( hist1,color="r",label="灰度图像直方图",linestyle="--")#, plt.imshow(hist1)
    #plt.title("灰度图像直方图")
    plt.legend()
    plt.subplot(122),
    plt.plot( hist2,color="b",label="高斯滤波直方图",linestyle="--")
    #plt.title("高斯滤波直方图")
    plt.legend()

    #plt.plot( hist,color="lime",label="高斯滤波直方图",linestyle="--")
    plt.show()