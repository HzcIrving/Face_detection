#-*- coding: utf-8 -*- 

# STEP1 
# 获取并显示USB摄像头实时视频
# 识别出人脸(OPENCV直接调用的API)

# STEP2		
# 准备训练数据，对图像进行预处理与标注
# 用CNN读取数据集

import cv2
import sys 
import numpy as np 
import os 
from PIL import Image

# GLOBAL VARIABLES
IMAGE_SIZE = 64
BLACK = [0,0,0] #RGB
IMAGES = [] #训练图像
LABELS = [] #训练图像的 label
PATH = 'F:\\Coursera\\FACEDETECTION_hzc\\datasets' #for small test 

def CatchUsbVedio(window_name, camera_id,catch_pic_num,path_name):
	cv2.namedWindow(window_name)

	#视频来源
	#1.保存的视频
	#2.网络摄像头

	cap = cv2.VideoCapture(camera_id)

	# 告诉OpenCV使用人脸分类器
	classifier = cv2.CascadeClassifier("F:/OPENCV/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml")

	# 人脸的anchor box的颜色
	color = (255,0,0)
	num = 0

	while cap.isOpened():
		ok,frame = cap.read() #读取一帧
		if not ok:
			break 

		#将图像转为灰度图像
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#人脸检测
		# ScaleFactor:缩放比例,同一物体，与相机距离不同，大小亦不相同，必须将其缩放到一定大小方便识别
		# minNeighbors:对特征点周边有多少有效点同时检测，可避免选取的特征点太小导致易漏
		# minSize:特征点的最小值
		FaceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
		if len(FaceRects)>0: #大于0检测出人脸
			#输出多张人脸
			for FaceRect in FaceRects: 	#单独框出每一个人的脸
				x,y,w,h = FaceRect

				#保存当前帧
				img_name = '%s%d.jpg'%(path_name,num)
				image = frame[y-10:y+h+10,x-10:x+w+10] #裁剪出人脸
				cv2.imwrite(img_name, image)

				num+=1

				if num>(catch_pic_num):
					break

				#画出矩形框
				cv2.rectangle(frame, (x-10,y-10), (x+w+10,y+h+10), (255,0,0), 2)
				#显示当前捕捉了多少张人脸
				font = cv2.FONT_HERSHEY_COMPLEX
				cv2.putText(frame,'num:%d'%(num),(x+10,y+30),font,1,(255,0,255),4)

		#超过最大保存数，结束程序
		if num > (catch_pic_num):
			break 

		#显示图像
		#等待10ms,按'q'键退出
		cv2.imshow(window_name, frame)
		c = cv2.waitKey(10)
		if c & 0xFF == ord('q'):
			break

	#释放摄像头并销毁所有窗口
	cap.release()
	cv2.destroyAllWindows()

# 调整图像尺寸大小  64x64加快学习
def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
	top,bottom,left,right=(0,0,0,0)

	#获取图像尺寸
	h,w, _ = image.shape 

	#长短不一的，找到最长边
	longest_edge = max(h,w)

	#计算短边需要增加的像素，使其与长边相等
	if h < longest_edge:
		dh = longest_edge - h 
		top = dh//2
		bottom = dh-top
	elif w < longest_edge:
		dw = longest_edge - w
		left = dw//2
		right = dw-left 
	else:
		pass

	# 给图像增加边界，使图像长宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
	# 给图像增加边界
	# https://www.cnblogs.com/pakfahome/p/3914318.html
	# BORDER_REFLICATE 直接用边界颜色填充
	# BORDER_REFLECT 倒映 
	# BORDER_REFLECT_101 倒映 倒映时，会空开边界
	# BORDER_WRAP 仿射
	# BORDER_CONSTANT 填充颜色，颜色值由value定，这里用的是BLACK=[0,0,0]
	# 比如200x300的>>64x64,需要填充黑边在上下
	constant = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)

	#调整图像大小并返回
	return cv2.resize(constant,(height,width))

images=[]
labels=[]

def read_path(path_name):
	for dir_item in os.listdir(path_name):
		# 当前目录下所有文件
		#从初始路径开始，合并成可识别的操作路径
		# os.path.join将多个路径组合后返回，第一个绝对路径之前的参数将被忽略。
		# full_path = os.path.abspath(os.path.join(path_name,dir_item))
		full_path = os.path.join(path_name,dir_item)
		# print(full_path) #当前路径

		#如果是文件夹，继续递归调用
		if os.path.isdir(full_path): #如果是文件夹，继续递归调用
			read_path(full_path) #递归
		else: #文件 
			if dir_item.endswith('.jpg'):
				image = cv2.imread(full_path)
				# print(image.shape)
				# cv2.imshow("test",image)
				# cv2.waitKey(10) #10ms 
				# cv2.destroyAllWindows()

				image = resize_image(image,IMAGE_SIZE,IMAGE_SIZE) #调用函数进行处理

				# 实际使用注释掉
				# cv2.imwrite('1.jpg',image)
				# cv2.imshow("test",image)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()

				images.append(image)
				labels.append(path_name)

	return images,labels

#从指定路径读取训练数据
def load_dataset(path_name):
	images,labels = read_path(path_name)
	# print("Before ...",images.shape)
	#将所有图片转为四维数组，尺寸为：(nums,image_size,image_size,channels)
	images = np.array(images)
	print('After ...',images.shape)
	# print(labels)
	#标注数据"HeZichen"是0, "Luna"是1
	labels = np.array([0 if label.endswith("HeZichen") else 1 for label in labels])
	return images,labels

# if __name__ == '__main__':


	# 0 就是你的laptop的摄像头
	# CatchUsbVedio("截取视频流", 0, 650, "datasets/HeZichen/")
	# test = cv2.imread(PATH)
	# cv2.imshow('image',test)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# images,labels = load_dataset(PATH)