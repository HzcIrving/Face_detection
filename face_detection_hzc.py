# -*- coding: utf-8 -*-

# 搭建训练模型>>>开跑啦...
import random 
import numpy as np 

from util_tools import *

from sklearn.cross_validation import train_test_split #分割测试训练集
# import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D 
from keras.optimizers import SGD
from keras.utils import np_utils 
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')

#GLOBAL VARIABLES
MODEL_PATH = 'F:\\Coursera\\FACEDETECTION_hzc\\Model\\me.face.model.h5'
IMAGE_SIZE = 64 #default

class Dataset:
	"""	
	1.一部分用于训练
	2.一部分用于测试
	3.一部分用于验证
	4.归一化与预处理操作	
	封装在Dataset类里面
	"""
	def __init__(self,path_name):
		#类的初始化

		#训练集
		self.train_images = None 
		self.train_labels = None 

		#测试集
		self.test_images = None 
		self.test_labels = None 

		#验证集 __ Cross Validation >>> params regulating
		self.valid_images = None 
		self.valid_labels = None 

		#数据集加载路径
		self.path_name = path_name

		#当前库采用的维度顺序
		self.input_shape = None 

	def load(self,img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, \
				nb_classes=2):
		"""
		1.加载数据集
		2.按照交叉验证规则划分
		3.预处理
		"""
		#加载数据到内存
		images,labels = load_dataset(self.path_name)

		#划分train和valid，比例0.3
		train_images,valid_images,train_labels,valid_labels = train_test_split(images,\
			labels,test_size=0.3,random_state = random.randint(0,100))
		#划分test和validation，比例0.3
		_,test_images,_,test_labels = train_test_split(images,labels,test_size=0.3,\
			random_state=random.randint(0,100))

		#当前的维度顺序如果"th"，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
		#这部分代码是根据Keras库的要求的维度顺序重组训练数据集
		#keras在tensorflow上的维度顺序不太一样...
		#th>>>theano  tf>>tensorflow
		if K.image_dim_ordering() == 'th':
			train_images = train_images.reshape(train_images.shape[0],img_channels,img_rows,img_cols)
			valid_images = valid_images.reshape(valid_images.shape[0],img_channels,img_rows,img_cols)
			test_images = test_images.reshape(test_images.shape[0],img_channels,img_rows,img_cols)
			self.input_shape = (img_channels,img_rows,img_cols)
		else:
			train_images = train_images.reshape(train_images.shape[0],img_rows,img_cols,img_channels)
			valid_images = valid_images.reshape(valid_images.shape[0],img_rows,img_cols,img_channels)
			test_images = test_images.reshape(test_images.shape[0],img_rows,img_cols,img_channels)
			self.input_shape = (img_channels,img_rows,img_cols)

		#输出训练集、验证集、测试集的数量
		print(train_images.shape[0],'train samples')
		print(test_images.shape[0],'test samples')
		print(valid_images.shape[0],'valid samples')

		#one-hot编码，方便softmax 向量化
		#将nb_classes >> 转为二维
		train_labels = np_utils.to_categorical(train_labels, num_classes=nb_classes)
		test_labels = np_utils.to_categorical(test_labels,num_classes=nb_classes)
		valid_labels = np_utils.to_categorical(valid_labels, num_classes=nb_classes)

		#数据浮点化，方便归一化
		train_images = train_images.astype('float32')
		test_images = test_images.astype('float32')
		valid_images = valid_images.astype('float32')

		#归一化0~1
		#大的值与小的值在一起MSE，会影响最终结果
		#权重平均化
		train_images /= 255
		valid_images /= 255
		test_images /= 255

		self.train_images = train_images 
		self.valid_images = valid_images
		self.test_images = test_images
		self.train_labels = train_labels
		self.valid_labels = valid_labels 
		self.test_labels = test_labels

#构建CNN模型
class Model:
	def __init__(self):
		self.model = None 

	#建立模型
	def build_model(self,dataset,nb_classes=2):
		#构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
		self.model = Sequential()

		#添加CNN网路各层
		#32个卷积核，每个卷积核大小为3x3，'same'保留边界特征的方式滑动，‘valid'是丢掉
		#input_shape应该是64x64x3
		self.model.add(Convolution2D(32,(3,3),border_mode='same',\
						input_shape = dataset.input_shape)) #卷积层
		self.model.add(Activation('relu'))

		self.model.add(Convolution2D(32,(3,3)))
		self.model.add(Activation('relu'))

		self.model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first")) #下采样池化
		self.model.add(Dropout(0.25)) #做Dropout，避免过拟合，加快学习速度

		self.model.add(Convolution2D(64,(3,3)))
		self.model.add(Activation('relu'))

		self.model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first")) #池化
		self.model.add(Dropout(0.25))

		#构建去连接层
		self.model.add(Flatten())
		self.model.add(Dense(512)) #全连接层
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(nb_classes)) #换成要输出的两类
		self.model.add(Activation('softmax')) #分类层

		#输出模型概况
		self.model.summary()

	def train(self,dataset,batch_size=20,nb_epoch=10,data_augmentation=True):
		#模型训练过程
		
		#优化器选择：SGD+MOMENTUM优化器进行训练
		sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)

		self.model.compile(loss='categorical_crossentropy',optimizer=sgd,
					metrics=['accuracy']) #完成实际的模型配置工作

		#是否使用数据增强
		if not data_augmentation:
			self.model.fit(dataset.train_images,
						   dataset.train_labels,
						   batch_size = batch_size,
						   np_epoch=nb_epoch,
						   validation_data=(dataset.valid_images,dataset.valid_labels),
						   shuffle=True)

		#使用数据增强加强模型泛化能力
		else:
			#定义数据生成器
			#用于做数据增强
			#返回一个生成器对象 datagenerator 
			#datagenerator每次被调用一次，其会生成一组数据（顺序生成）
			#节省内存，就是Python生成器
			datagenerator = ImageDataGenerator(
				featurewise_center=False,  #数据中心化开关 (均值0方差1)
				samplewise_center=False,  #是否使输入数据每个样本均值为0
				featurewise_std_normalization=False,  #是否数据标准化（输入数据除以数据集的标准差)
				samplewise_std_normalization=False,  #是否将每个样本数据除以自身的标准差
				zca_whitening=False,  #是否对输入数据ZCA白化
				zca_epsilon=1e-6, 
				rotation_range=20., #数据提升 图片随机转动的角度(0~180) 
				width_shift_range=0.2, #数据提升 图片水平平移的幅度 水平方向 
				height_shift_range=0.,  #同上 垂直方向
				shear_range=0., 
				zoom_range=0., 
				channel_shift_range=0., 
				fill_mode='nearest', 
				cval=0., 
				horizontal_flip=True, #是否进行随机水平翻转
				vertical_flip=False, #是否进行随机垂直翻转
				rescale=None, 
				preprocessing_function=None, 
				data_format=None)

			#计算整个训练样本集的数量，用于特征值归一化，ZCA白化
			print(type(dataset.train_images))
			datagenerator.fit(dataset.train_images)

			#利用生成器开始训练模型
			self.model.fit_generator(datagenerator.flow(dataset.train_images,
										dataset.train_labels,
										batch_size=batch_size),
									samples_per_epoch=dataset.train_images.shape[0],
									nb_epoch=nb_epoch,
									validation_data=(dataset.valid_images,dataset.valid_labels))

	def evaluate(self,dataset):
		#提供测试集模型
		#用测试集验证模型
		score = self.model.evaluate(dataset.test_images,dataset.test_labels,verbose=1)
		print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

	def save_model(self,file_path=MODEL_PATH):
		#模型保存
		self.model.save(file_path)

	def load_model(self,file_path=MODEL_PATH):
		#模型读取
		self.model = load_model(file_path)

	def predict_face(self,image):
		#做图片预测
		#dimension:1,IMAGE_SIZE,IMAGE_SIZE,3
		#维度选择
		#该模块供外部调用
		if K.image_dim_ordering() == "th" and image.shape != (1,3,IMAGE_SIZE,IMAGE_SIZE):
			image = resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE)
			image = image.reshape((1,3,IMAGE_SIZE,IMAGE_SIZE))
		elif K.image_dim_ordering() == "tf" and image.shape != (1,IMAGE_SIZE,IMAGE_SIZE,3):
			image = resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE)
			image = image.reshape((1,IMAGE_SIZE,IMAGE_SIZE,3))

		#浮点归一化
		image = image.astype('float32')
		image /= 255

		#给出输入属于各个类别的概率，二值类别，函数会给出输入图像属于0和1的概率各为多少
		result = self.model.predict_proba(image)
		print('result:', result)

		#给出类别预测 0或1
		result = self.model.predict_classes(image)

		#返回类别预测结果
		return result[0]


# if __name__  == '__main__':

# 	dataset = Dataset(PATH)
# 	dataset.load(img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2)

# 	# model = Model()
# 	# model.build_model(dataset)
# 	# #测试训练函数代码
# 	# model.train(dataset)
# 	# model.save_model(file_path='F:\\Coursera\\FACEDETECTION_hzc\\Model\\me.face.model.h5')
# 	# model.save_model()

# 	#评估模型
# 	model = Model()
# 	model.load_model(file_path=MODEL_PATH)
# 	model.evaluate(dataset)

####
#错误1:注意theano和tensorflow的格式问题
	# tf:(m,width,height,channels)
	# th:(m,channels,width,height)
	# 之前：K.set_image_dim_ordering('th')





