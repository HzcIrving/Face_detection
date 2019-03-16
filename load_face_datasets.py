#-*- coding: utf-8 -*-

# 识别出人脸
# OpenCV有现成的API可以快速对人脸进行识别
# 训练数据准备好，并将其输入给CNN。
# 前面我们已经准备好了2000张脸部图像，但没有进行标注，并且还需要将数据加载到内存，以方便输入给CNN。
# 因此，第一步工作就是加载并标注数据到内存。

import os 
import cv2 
import numpy as np 
import sys 

# 全局变量 统一图片大小
IMAGE_SIZE=64

# 按照指定图像大小调节尺寸
def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
	top,bottom,left,right=(0,0,0,0)

	#获取图像尺寸
	h,w,_ = image.shape #H,W,C

	#对于长宽不相等的图片，找到最长的一边
	longest_edge = max(h,w)

	#计算短边需要增加多上像宽度使其与长边相等
	if h < longest_edge:
		dh = longest_edge - h
		top = dh // 2
		bottom = dh - top
	elif w <longest_edge:
		dw = longest_edge - w 
		left = dw // 2 
		right  = dw - left 
	else:
		pass

	#RGB颜色
	BLACK = [0,0,0]

	#给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
	#边界为黑色
	constant = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)	

	#调整图像大小并返回
	return cv2.resize(constant,(height,width))

#读取训练数据
images = []
labels = []
def read_path(path_name):
	for dir_item in os.listdir(path_name):
		#从初始路径开始叠加，合并成可识别的操作路径
		full_path = os.path.abspath(os.path.join(path_name,dir_item))

		if os.path.isdir(full_path): # 如果是文件夹，继续递归调用
			read_path(full_path)
		else:  # 是文件
			if dir_item.endswith('.jpg'): #如果是以jpg格式结尾的
				image = cv2.imread(full_path)
				image = resize_image(image,IMAGE_SIZE,IMAGE_SIZE) #预处理

				# 测试一下resize_image的效果
				# cv2.imwrite('1.jpg',image)

				images.append(image)
				labels.append(path_name) #添加label 一般是名字..

	return images, labels

#从指定路径读取训练数据
def load_dataset(path_name):
	images, labels = read_path(path_name)

	#将所有图片转为 (图片数量,image_size,image_size,3)>>> (1000,64,64,3)
	


