import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from matplotlib import pyplot as plt
from cutImage import cutImage


def initGetRGBDFrame():
    #创建一个管道
    pipeline = rs.pipeline()
    #Create a config并配置要流​​式传输的管道
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)#1024，768
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)#1920，1080
    #开始流式传输
    profile = pipeline.start(config)
    #获取深度传感器的深度标尺 #已经测量
    #depth_sensor = profile.get_device().first_depth_sensor()
    #depth_scale = depth_sensor.get_depth_scale()
    #print("Depth Scale is: " , depth_scale)
    #创建对齐对象
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline,align

def getRGBDFrame(pipeline,align):
    #获取颜色和深度的框架集
    frames = pipeline.wait_for_frames()
    #对齐深度与颜色
    aligned_frames = align.process(frames)
    #获取对齐的帧
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    #帧无效则重新获取
    if not aligned_depth_frame or not color_frame:
        return getRGBDFrame(pipeline,align)
    #深度数据（还未用到）    
    #depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")    
    #获取cv格式的图像
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image,depth_image



if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #初始化
    pipeline,align = initGetRGBDFrame()

    #对彩色图和深度图进行处理
    while True:
        #获取彩色图和深度图
        color_image,depth_image = getRGBDFrame(pipeline,align)
        cutImage(color_image,depth_image)
        cv2.waitKey(10)
    

    ##删除clipping_distance_in_meters外的背景
    #clipping_distance_in_meters = 1 #可以修改识别的距离 1 meter
    ##删除背景
    #bg_removed=removeBackground(color_image,depth_image, clipping_distance_in_meters )

    #plt.subplot(231),plt.imshow(color_image)
    #plt.title("Depth"),plt.xticks([]),plt.yticks([]) 
    #plt.subplot(232),plt.imshow(bg_removed)
    #plt.title("RmBackground"),plt.xticks([]),plt.yticks([])
    #depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    #plt.subplot(233),plt.imshow(depth_mapped_image)
    #plt.title("Depth"),plt.xticks([]),plt.yticks([]) 
    #plt.show()

    #depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    #cv2.imshow("live", np.hstack((color_image, depth_mapped_image)))
    #key = cv2.waitKey(0)

