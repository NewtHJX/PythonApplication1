import cv2
import numpy as np

def test1():
    #图片的读取与显示
    imageDir = "image.jpg"
    #flag = 1
    src = cv2.imread(imageDir,0)#flag图片读取方式0灰度 1彩色
    cv2.namedWindow("test_windows",0)#0可以改变窗口大小 不写则不能改变
    cv2.imshow("test_windows",src)
    cv2.waitKey(0) #0一直显示，直到按下数字
    cv2.destroyAllWindows()

def test2():
    #图片的各种参数
    imageDir = "image4.jpg"
    src = cv2.imread(imageDir,1)
    cv2.imshow("image",src)
    print(type(src))#<class 'numpy.ndarray'>
    print(src.shape)#(828, 903, 3)
    print(src.size)#2243052
    print(src.dtype)#uint8
    pixel_data = np.array(src)#数组表示
    print(pixel_data)
    cv2.imwrite("image_ref.png",src)#图片的写入
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test3():
    #元组 列表 区间 的各种操作
    a_tuple = ('abc',666,'hjx');print(a_tuple);print(type(a_tuple[1]))
    a_list = list(a_tuple);print(a_list)
    a_range = range(1,10,2);print(a_range)
    b_range = range(10);print(b_range)
    b_list = list(a_range);print(b_list)
    b_tuple = tuple(range(1,20,3));print(b_tuple)
    a_list.append(a_tuple);print(a_list);print(type(a_list))
    c_list = list(range(1,9,3))
    c_list.extend(a_list);print(c_list)
    a_list.extend('aaa');print(a_list)
    a_list.insert(2,'aaa');print(a_list)
    c_list.clear()
    a_list.remove('aaa');print(a_list)
    a_list.pop();print(a_list)



#test1();
#test2();
test3();

