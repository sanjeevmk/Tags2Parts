import random
from keras.models import Model
import tensorflow as tf
random.seed(10)
import numpy as np
import sys
from keras.layers.convolutional import UpSampling3D,ZeroPadding3D,UpSampling2D
from keras.layers import Lambda,Activation,merge,Dense,BatchNormalization,Dropout
from . layers import input_layer,output_layer,Convolution3DRelu,MaxPooling3D,Flatten,AvgPooling3D,AvgPooling3DStrides,maxnorm,maxnormshape,phase_trn,add,lse,WX,Convolution3DSig,DeConvolution3DRelu,DenseRelu,DeConvolution2DRelu,Convolution2DRelu,Convolution2DTanh
from keras import backend as K
from keras.layers.advanced_activations import ThresholdedReLU
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from . import network_utils
import datetime
import os,sys,inspect
import csv
from keras import regularizers
from . import tf_util
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_bn_decay(batch,batch_size):
        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch*batch_size,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

def cross_entropy(true,pred):
	ce = -tf.reduce_sum( (  (true*tf.log(pred+ 1e-9)) + ((1-true) * tf.log(1 - pred+ 1e-9)) )  )
	return ce

class Network:
	def __init__(self):
		self.sess = tf.Session()
		self.model = None

	def local_initializer(self):
		self.sess.run(tf.local_variables_initializer())

	def initializer(self):
		self.sess.run(tf.global_variables_initializer())

	def check_and_initialize(self):
		init_list = []
		for var in tf.all_variables():
			try:
				self.sess.run(var)
			except tf.errors.FailedPreconditionError:
				init_list.append(var)

		self.varlist_initializer(init_list)

	def varlist_initializer(self,var_list):
		self.sess.run(tf.variables_initializer(var_list=var_list))

	def varlist_optimizer(self,losses_dict,var_list,lr=None):
		error = 0.0
		for key in losses_dict:
			error += losses_dict[key]
		if lr is None:
			trainer = tf.train.AdamOptimizer(1e-3).minimize(error,var_list=var_list)
		else:
			trainer = tf.train.AdamOptimizer(lr).minimize(error,var_list=var_list)
		#trainer = tf.train.MomentumOptimizer(1e-3,0.9,use_nesterov=True).minimize(error,var_list=var_list)
		init_list = []
		for var in tf.all_variables():
			try:
				self.sess.run(var)
			except tf.errors.FailedPreconditionError:
				init_list.append(var)

		self.varlist_initializer(init_list)
		return error,trainer

	def optimizer(self,losses_dict,train=True):
		error = 0.0
		for key in losses_dict:
			error += losses_dict[key]
		if train:
			trainer = tf.train.AdamOptimizer(1e-3).minimize(error)
		else:
			trainer = None

		return error,trainer
			
	def load(self,modelpath):
		tfsaver  = tf.train.Saver(max_to_keep=None)
		tfsaver.restore(self.sess,modelpath)

class Architectures(Network):
    def wunet(self,input_pl,batch,is_training_pl,_bn_decay):
        with tf.device('/gpu:0'):
                conv5x5_1 = Convolution3DRelu(input_pl,1,5,5,5,4,init='glorot_uniform',name='c1',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_1_s = Convolution3DRelu(conv5x5_1,4,5,5,5,4,init='glorot_uniform',name='c2',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_1_p = MaxPooling3D(conv5x5_1_s,2,2,2)
        with tf.device('/gpu:0'):
                conv5x5_6 = Convolution3DRelu(conv5x5_1_p,4,5,5,5,8,init='glorot_uniform',name='c11',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_6_s = Convolution3DRelu(conv5x5_6,8,5,5,5,8,init='glorot_uniform',name='c12',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_6_u = tf.concat([UpSampling3D((2,2,2),data_format='channels_last')(conv5x5_6_s),conv5x5_1_s],4)

        with tf.device('/gpu:0'):
                conv5x5_7 = Convolution3DRelu(conv5x5_6_u,12,5,5,5,4,init='glorot_uniform',name='c13',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_7_s = Convolution3DRelu(conv5x5_7,4,5,5,5,4,init='glorot_uniform',name='c14',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_7_p = tf.concat([MaxPooling3D(conv5x5_7_s,2,2,2),conv5x5_6_s],4)
        with tf.device('/gpu:0'):
                conv5x5_9 = Convolution3DRelu(conv5x5_7_p,12,5,5,5,8,init='glorot_uniform',name='c17',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_9_s = Convolution3DRelu(conv5x5_9,8,5,5,5,8,init='glorot_uniform',name='c18',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_9_u = tf.concat([UpSampling3D((2,2,2),data_format='channels_last')(conv5x5_9_s),conv5x5_7_s],4)

        with tf.device('/gpu:0'):
                conv5x5_10 = Convolution3DRelu(conv5x5_9_u,12,5,5,5,4,init='glorot_uniform',name='c19',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_10_s = Convolution3DRelu(conv5x5_10,4,5,5,5,4,init='glorot_uniform',name='c20',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_10_p = tf.concat([MaxPooling3D(conv5x5_10_s,2,2,2),conv5x5_9_s],4)
        with tf.device('/gpu:0'):
                conv5x5_11 = Convolution3DRelu(conv5x5_10_p,12,5,5,5,8,init='glorot_uniform',name='c21',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_11_s = Convolution3DRelu(conv5x5_11,8,5,5,5,8,init='glorot_uniform',name='c22',batchnorm=True,istraining=is_training_pl,bn_decay=_bn_decay)
                conv5x5_11_u = tf.concat([UpSampling3D((2,2,2),data_format='channels_last')(conv5x5_11_s),conv5x5_10_s],4)

        return conv5x5_11_u,12 
