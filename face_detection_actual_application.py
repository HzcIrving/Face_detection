#-*- coding: utf-8 -*-

import cv2
import sys 
import gc
from face_detection_hzc import Model

#GLOBAL VARIABLES
MODEL_PATH = 'F:\\Coursera\\FACEDETECTION_hzc\\Model\\me.face.model.h5'
IMAGE_SIZE = 64 #default

if __name__ == '__main__':

	#加载模型
	model = Model()
	model.load_model(file_path=MODEL_PATH)

	#人脸anchor box颜色
	color = (0,255,255)

	#捕获指定摄像头实时视频流
	cap = cv2.VideoCapture(0)

	#告诉OpenCV使用人脸分类器地址
	cascade_path = "F:/OPENCV/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml"

	#人脸循环检测
	while  True:
		_,frame = cap.read() #读取一帧

		#图像灰化，降低计算复杂度
		frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		#使用人脸检测
		cascade = cv2.CascadeClassifier(cascade_path)

		#利用分类器识别出哪个区域是人脸
		faceRects = cascade.detectMultiScale(frame_gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
		if len(faceRects) > 0:
			for faceRect in faceRects:
				x,y,w,h = faceRect

				#截取脸部提交给之前训练的卷积神经网络模型进行识别
				image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
				faceID = model.predict_face(image)

				#如果是“我”
				if faceID == 1:

					cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

					#文字提醒
					cv2.putText(frame,'HeZichen',
								(x+30,y+30),    		 	 #坐标
								 cv2.FONT_HERSHEY_COMPLEX,   #字体
								 1,							 #字号
								 (255,255,255),				 #颜色 
								 2)							 #线宽
				else:
					pass 

		cv2.imshow("识别何子辰", frame)

		#等待10毫秒看是否有按键输入
		k = cv2.waitKey(10)
		#如果输入q则退出循环
		if k & 0xFF == ord('q'):
			break

	#释放摄像头并销毁所有窗口
	cap.release()
	cv2.destroyAllWindows()						