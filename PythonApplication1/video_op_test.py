import numpy as np
import cv2
import time

#from numpy.core.arrayprint import printoptions
#from geometry_msgs.msg import Twist
#import rospy

def cap_video():
    cap = cv2.VideoCapture(0) #��ͨ��0Ϊusb����ͷ #�Ľ���VideoCapture���Կ���֡���𣿶�ʵ��ʱ���кܴ�Ӱ����
    if cap.isOpened():
        print("Camera is on\n")
        show_video(cap)
    else:
        print("Fail to open camera\n")

def show_video(cap):
    while 1:
        ret,frame = cap.read()
        if ret:
            # time.clock()
            cut_image(frame)
            if cv2.waitKey(10) & 0xff == 27: break #27 ��ESC�˳���ord('q')��Q�˳�
        else :
            print("There is a missing frame\n")

def cut_image(sourceDir):

    # ��ȡͼƬ
    img = sourceDir
    #img_shape = img.shape

    global height #�Ľ���ʱû���õ���ͼƬ�ĸߺͿ� #480 640
    height = img.shape[0]
    global width 
    width = img.shape[1];
    #print(height,width)

    # �ҶȻ�
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ��˹ģ������:ȥ��
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    # Sobel����XY�����ݶ�
    gradX = cv2.Sobel(blur, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blur, ddepth=cv2.CV_32F, dx=0, dy=1)
    # �����ݶȲ�
    gradient = cv2.subtract(gradX, gradY)
    #����ֵ
    gradient = cv2.convertScaleAbs(gradient)
    #��˹ģ������ȥ��
    blured = cv2.GaussianBlur(gradient, (9, 9), 0)

    # ��ֵ�� #��ֵ����ͼ��ͷ��ֵ�ͼ���ཻ
    _ , dst = cv2.threshold(blured, 20, 255, cv2.THRESH_BINARY) #30Ϊ�ֽ� #�Ľ���Ҫ�ҳ���ֵ��ֵ����ʱ��ȡ�����Ч��
    cv2.imshow("Dst", dst)#��ֵ����ͼ��
    
    mask=minus_hand(img)
    #cv2.imshow("mask",mask)
    dst = cv2.bitwise_xor(dst,mask)
    cv2.imshow("Dst2", dst)#��ֵ����ͼ��

    # �ṹԪ��
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,45)) #�Ľ���һ������ 107,76 ���ǰ����ų����⣬�Ϳ��Ը������ͨ
    # ��̬ѧ����:��̬�մ���(��ʴ)
    closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Closed", closed)#��һ�αղ���
    # ��ʴ�����͵���
    closed = cv2.erode(closed, None, iterations=3)
    closed = cv2.dilate(closed, None, iterations=3)
    cv2.imshow("Closed2", closed)#��̬ѧ������ͼ��
    # ��ȡ����,
    cnts,he= cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(cnts))
    try:
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    except:
        return
    #�����С������
    rect = cv2.minAreaRect(c)
    #print(rect[0])#center of rect
    cmd = displacement(rect)
    # print(cmd)
    box = np.int0(cv2.boxPoints(rect))

    draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
    cv2.imshow("Box", draw_img)

def displacement(rect):
    # put the data in a list
    pub = []
    #pub.extend(rect);print(type(pub));print(pub)
    cmd = []
    for item in rect:
        if isinstance(item, tuple):
            for x in item:
                pub.append(x)
        else:
            pub.append(item)
    global pub_pre,ret_x,ret_y
    if len(pub_pre):
        cmd = [a - b for a, b in zip(pub, pub_pre)] #����λ��
        x = cmd[0]
        y = cmd[1]
        ret_x,avg_x = avg_filter(x,ret_x)
        ret_y,avg_y = avg_filter(y,ret_y)
        if (avg_y!=-1)&(avg_x!=-1):
            cmd_pub = [avg_x,avg_y]
            publ(cmd_pub)
            # print (cmd_pub)
    pub_pre = pub
    return cmd

def publ(m_cmd):
    #move_cmd = getZeroTwist()
    global height,width
    #print(m_cmd)
    distance = 1e-5
    x=int(m_cmd[0]*100)
    y=int(m_cmd[1]*100)
    xi = np.abs(x)
    #yi = np.abs(y)/xi
    # print(xi)
    for i in range(1,101): #�Ľ� ������ʵ�ַ�ʽ
        #xi -=1
        ma= m_cmd[0]*distance
        mb = m_cmd[1]*distance
        #print(ma,mb)
        #move_cmd.linear.x = np.sign(x)*distance
        #move_cmd.linear.y = yi*np.sign(y)*distance
        # print(move_cmd.linear.x)
        # print(move_cmd.linear.y)
        #vel_pub.publish(move_cmd)

def getZeroTwist():
    #��ʼ��
    t = Twist()
    t.linear.x = 0
    t.linear.y = 0
    t.linear.z = 0
    t.angular.x = 0
    t.angular.y = 0
    t.angular.z = 0
    return t

def avg_filter(x,ret,length=5):
    avg = -1
    if (len(ret)<length):        
        ret.append(x)
    else:
        avg = np.sum(ret)/length
        ret.append(x)
        ret = ret[-length:]
        #print(ret)
    return ret,avg

def minus_hand(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hand = np.array([0, 40, 0])#np.array([9, 40, 0])
    high_hand = np.array([43, 255, 255])#np.array([43, 255, 255])
    mask = cv2.inRange(hsv, low_hand, high_hand)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.bitwise_xor(gray,mask)
    cv2.imshow("test",dst)

    return mask


#vel_pub = rospy.Publisher('/cmd/Twist', Twist, queue_size=1)
#rospy.init_node('cmd_node') #��ʼ���ڵ� cmd_node

height = 0;width = 0
pub_pre = [] #
ret_x = []
ret_y = []
cap_video()
