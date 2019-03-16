#-*- coding: utf-8 -*-

# 获取并显示USB摄像头实时视频
# 识别出人脸(OpenCV有现成的API直接调用)

import cv2
import sys 
from PIL import Image

def CatchUsbVedio(window_name, camera_id,catch_pic_num,path_name):
	cv2.namedWindow(window_name)

	#视频来源
	#1.保存的视频
	#2.USB网络摄像头
	cap = cv2.VideoCapture(camera_id)

	#告诉OpenCV使用人脸识别分类器
	classifier = cv2.CascadeClassifier("F:/OPENCV/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml")

	#人脸anchor box颜色
	color = (255,0,0)

	num = 0 
	
	while cap.isOpened():
		ok,frame = cap.read() #读取一帧数据
		if not ok:
			break

		#将当前图像帧转换为灰度图像
		grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		#人脸检测
		#scaleFactor:图像缩放比例，同一物体，与相机距离不同，大小亦不相同，必须将其缩放到一定大小方便识别
		#minNeighbors:对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏
		#minSize:特征检测点的最小值
		FaceRects = classifier.detectMultiScale(grey,scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
		if len(FaceRects)>0: #大于0则检测出人脸
			#输出多张人脸
			for FaceRect in FaceRects:  #单独框处每一张人脸
				x,y,w,h = FaceRect

				#保存当前帧
				img_name = '%s%d.jpg'%(path_name,num)
				image = frame[y-10:y+h+10,x-10:x+w+10] #剪裁出脸部
				cv2.imwrite(img_name,image)

				num+=1
				if num>(catch_pic_num):
					break 

				#画出矩形框
				# cv2.rectangle()
				cv2.rectangle(frame, (x-10,y-10), (x+w+10,y+h+10),(255,0,0),2)

				#显示当前捕捉到了多少张人脸图片
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame,'num:%d' %(num),(x+30,y+30),font,1,(255,0,255),4)

		#超过指定最大保存数就结束程序
		if num>(catch_pic_num):
			break 

		#显示图像
		#等待10ms 按键输入: 'q'exit..
		cv2.imshow(window_name,frame)
		c = cv2.waitKey(10)
		if c & 0xFF == ord('q'):
			break

	#释放摄像头并销毁所有窗口
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':

	# sys.argv是获取运行python文件的时候命令行参数
	# print(sys.argv[0])
	# if len(sys.argv) != 2:
	# 	print("Usage:%s camera_id\r\n" %sys.argv[0])
	# else:
	# 	CatchUsbVedio("截取视频流",int(sys.argv[1]))
	CatchUsbVedio("截取视频流",0,1000,"datasets/HeZichen/")
