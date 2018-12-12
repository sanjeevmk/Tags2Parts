import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import control_flow_ops
import random
import numpy as np
from . import tf_util

def zero_weight_variable(shape,name):
  initial = tf.zeros(shape,dtype='float32')
  return tf.get_variable(name,initializer=initial)

def zero_bias_variable(shape,name):
  initial = tf.zeros(shape,dtype='float32')
  return tf.get_variable(name,initializer=initial)

def zero_untrain(shape,name):
  initial = tf.zeros(shape,dtype='float32')
  return tf.get_variable(name,initializer=initial,trainable=False)

def trunc_normal_weight_variable(shape):
	#value = tf.random_uniform_initializer(-5, 5, dtype='float32', seed=10)(shape)
	#return tf.Variable(value, dtype='float32')
	initial = tf.truncated_normal(shape,stddev=0.01,dtype='float32')
	return tf.Variable(initial)

def uniform_weight_variable(shape,name,seed=10):
	initial = tf.random_uniform(shape, minval=-0.05,maxval=0.05,dtype='float32',seed=seed)
	return tf.get_variable(name,initializer=initial)

def random_conv(shape,name,seed=10):
	arr = np.zeros(shape,dtype='float32')
	rxs = random.sample(range(shape[0]),int(shape[0]/2))
	rys = random.sample(range(shape[1]),int(shape[1]/2))
	rzs = random.sample(range(shape[2]),int(shape[2]/2))

	arr[rxs,rys,rzs,:,:] = 1
	init = tf.constant(arr)
	return tf.get_variable(name, initializer=init,trainable=False)

def uniform_weight_variable_big(shape,name,seed=10):
	initial = tf.random_uniform(shape, minval=-5,maxval=5,dtype='float32',seed=seed)
	return tf.get_variable(name,initializer=initial)

def glorot_weight_variable(shape,name,seed=10):
	initial = tf.get_variable( name,shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True,seed=seed) )
	return initial

def bias_variable(shape,name):
	initial = tf.constant(0.0,shape=shape)
	return tf.get_variable( name, initializer=initial)
	#return initial

def glorot_bias_variable(shape,name):
	initial = tf.get_variable( name,shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,seed=10) )
	return initial

def tf_conv2d(x, W,padding='SAME',strides=[1,1,1,1]):
	return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def tf_conv3d(x, W,padding='SAME',strides=[1,1,1,1,1]):
	return tf.nn.conv3d(x, W, strides=strides, padding=padding)

def tf_conv3d_t(x, W,oshape,padding='SAME',strides=[1,2,2,2,1]):
	return tf.nn.conv3d_transpose(x, W, oshape,strides=strides, padding=padding)

def tf_conv2d_t(x, W,oshape,padding='SAME',strides=[1,1,1,1]):
	return tf.nn.conv2d_transpose(x, W, oshape,strides=strides, padding=padding)

def tf_max_pool(x,dim1,dim2,dim3):
  return tf.nn.max_pool3d(x, ksize=[1, dim1, dim2, dim3,1],strides=[1, dim1, dim2, dim3,1], padding='SAME')

def tf_avg_pool(x,dim1,dim2,dim3):
  return tf.nn.avg_pool3d(x, ksize=[1, dim1, dim2, dim3,1],strides=[1, dim1, dim2, dim3,1], padding='SAME')

def tf_avg_pool_stride(x,dim1,dim2,dim3,dim1s,dim2s,dim3s):
  return tf.nn.avg_pool3d(x, ksize=[1, dim1, dim2, dim3,1],strides=[1, dim1s, dim2s, dim3s,1], padding='SAME')

def phase_trn():
	return tf.placeholder(tf.bool,name='phase_train')

def input_layer(shape):
	return tf.placeholder(tf.float32, shape=shape)

def output_layer(numout):
	return tf.placeholder(tf.float32, shape=[None, numout])

def WX(inputtensor,in_channels,f_d,f_h,f_w,out_channels,init='uniform',name='',batchnorm=False,istraining=None,seed=10,padding='SAME'):
	W = None
	if init=='zero':
		W = zero_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] )

	if init=='uniform':
		W = uniform_weight_variable( [f_d,f_h,f_w,in_channels,out_channels],name,seed=seed )
		b = bias_variable( [out_channels],name+"_b" )

	if init=='glorot_uniform':
		W = glorot_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] ,name,seed=seed)

	if not batchnorm:
		return tf_conv3d(inputtensor,W,padding=padding) + b
	else:
		normed = batch_norm( tf_conv3d(inputtensor,W,padding=padding) , out_channels,istraining)
		return tf.nn.relu(normed)
	
def Convolution3DSig(inputtensor,in_channels,f_d,f_h,f_w,out_channels,init='uniform',name='',batchnorm=False,istraining=None,seed=10,padding='SAME',strides=[1,1,1,1,1],bn_decay=None):
	W = None
	b = None
	if init=='zero':
		W = zero_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] )
		b = zero_bias_variable( [out_channels] )

	if init=='uniform':
		W = uniform_weight_variable( [f_d,f_h,f_w,in_channels,out_channels],name,seed=seed )
		b = bias_variable( [out_channels],name+"_b" )

	if init=='glorot_uniform':
		W = glorot_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] ,name,seed=seed)
		b = bias_variable( [out_channels],name+'_b' )

	if init=='trunc_normal':
		W = trunc_normal_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] ,name,seed=seed)
		b = bias_variable( [out_channels] )

	convolution = tf_conv3d(inputtensor,W,padding=padding,strides=strides)  + b 

	if batchnorm:
		convolution = tf_util.batch_norm_template(convolution, istraining, name, [0,1,2,3], bn_decay)
	return tf.sigmoid( convolution )

	#return tf.sigmoid( tf_conv3d(inputtensor,W,padding=padding,strides=strides) + b )
	
def DeConvolution2DRelu(inputtensor,in_channels,f_h,f_w,out_channels,oshape,init='glorot_uniform',strides=[1,1,1,1],name='',batchnorm=False,is_training=None,bn_decay=None,seed=10,padding='SAME',reuse=False):
	W = None
	b = None
	with tf.variable_scope(name,reuse=reuse) as scope:
		if init=='glorot_uniform':
			W = glorot_weight_variable( [f_h,f_w,out_channels,in_channels],name,seed=seed )
			b = bias_variable( [out_channels], name+"_b")

		deconvolution = tf_conv2d_t(inputtensor,W,oshape,padding=padding,strides=strides)  + b
		if batchnorm:
			deconvolution = tf_util.batch_norm_template(deconvolution, is_training, name, [0,1,2], bn_decay)
		return tf.nn.relu(deconvolution)

def DeConvolution3DRelu(inputtensor,in_channels,f_d,f_h,f_w,out_channels,oshape,init='glorot_uniform',name='',batchnorm=False,istraining=None,seed=10,padding='SAME',bn_decay=None,strides=[1,2,2,2,1],activation_fn=None):
	W = None
	b = None
	if init=='zero':
		W = zero_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] )
		b = zero_bias_variable( [out_channels] )

	if init=='uniform':
		W = uniform_weight_variable( [f_d,f_h,f_w,out_channels,in_channels],name,seed=seed )
		b = bias_variable( [out_channels],name+"_b" )

	if init=='glorot_uniform':
		W = glorot_weight_variable( [f_d,f_h,f_w,out_channels,in_channels] ,name,seed=seed)
		b = bias_variable( [out_channels] , name+"_b" )

	if init=='trunc_normal':
		W = trunc_normal_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] )
		b = bias_variable( [out_channels] )
	
	deconvolution = tf_conv3d_t(inputtensor,W,oshape,padding=padding,strides=strides)  + b
	if batchnorm:
		deconvolution = tf_util.batch_norm_template(deconvolution, istraining, name, [0,1,2,3], bn_decay)

	if activation_fn is None:
		return tf.nn.relu(deconvolution)
	else:
		return activation_fn(deconvolution)

def DenseRelu(inputtensor,in_channels,out_channels,init='glorot_uniform',name='',batchnorm=False,istraining=None,seed=10,padding='SAME',strides=[1,1,1,1,1],wd=None,rw=False,bn_decay=None):
	W = None
	b = None

	if init=='glorot_uniform':
		W = glorot_weight_variable( [in_channels,out_channels] ,name,seed=seed)
		b = bias_variable( [out_channels] , name+'_b')
	if init=='trunc_normal':
		W = trunc_normal_weight_variable( [in_channels,out_channels])
		b = bias_variable( [out_channels], name+'_b' )

	wxb = tf.nn.bias_add(tf.matmul(inputtensor,W),b)
        #wxb = tf.nn.relu(wxb)
	if batchnorm:
		wxb = tf_util.batch_norm_template(wxb, istraining, name, [0,], bn_decay)
	return wxb
	#return tf.nn.relu(wxb)

def Convolution2DTanh(inputtensor,in_channels,f_h,f_w,out_channels,init='uniform',name='',batchnorm=False,istraining=None,seed=10,padding='SAME',strides=[1,1,1,1]):
	W = None
	b = None
	if init=='glorot_uniform':
		W = glorot_weight_variable( [f_h,f_w,in_channels,out_channels] ,name,seed=seed)
		b = bias_variable( [out_channels] , name+"_b" )
	if init=='trunc_normal':
		W = trunc_normal_weight_variable( [f_h,f_w,in_channels,out_channels] )
		b = bias_variable( [out_channels] , name+"_b")

	return tf.tanh( tf_conv2d(inputtensor,W,padding=padding,strides=strides)  + b )
	
def Convolution2DRelu(inputtensor,in_channels,f_h,f_w,out_channels,init='uniform',name='',batchnorm=False,istraining=None,seed=10,padding='SAME',strides=[1,1,1,1]):
	W = None
	b = None
	if init=='glorot_uniform':
		W = glorot_weight_variable( [f_h,f_w,in_channels,out_channels] ,name,seed=seed)
		b = bias_variable( [out_channels] , name+"_b" )
	if init=='trunc_normal':
		W = trunc_normal_weight_variable( [f_h,f_w,in_channels,out_channels] )
		b = bias_variable( [out_channels],name+"_b" )

	return tf.nn.relu( tf_conv2d(inputtensor,W,padding=padding,strides=strides)  + b )
	
def Convolution3DRelu(inputtensor,in_channels,f_d,f_h,f_w,out_channels,init='uniform',name='',batchnorm=False,istraining=None,seed=10,padding='SAME',strides=[1,1,1,1,1],bn_decay=None,activation_fn=tf.nn.relu):
	W = None
	b = None
	if init=='zero':
		W = zero_weight_variable( [f_d,f_h,f_w,in_channels,out_channels],name )
		b = zero_bias_variable( [out_channels],name+"_b" )

	if init=='uniform':
		W = uniform_weight_variable( [f_d,f_h,f_w,in_channels,out_channels],name,seed=seed )
		b = bias_variable( [out_channels],name+"_b" )

	if init=='uniformbig':
		W = uniform_weight_variable_big( [f_d,f_h,f_w,in_channels,out_channels],name,seed=seed )
		b = bias_variable( [out_channels],name+"_b" )

	if init=='glorot_uniform':
		W = glorot_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] ,name,seed=seed)
		b = bias_variable( [out_channels] , name+"_b" )

	if init=='trunc_normal':
		W = trunc_normal_weight_variable( [f_d,f_h,f_w,in_channels,out_channels] )
		b = bias_variable( [out_channels],name+"_b" )

	if init=='custom':
		W = random_conv( [f_d,f_h,f_w,in_channels,out_channels],name,seed=seed )
		b = zero_untrain( [out_channels],name+"_b" )

	convolution = tf_conv3d(inputtensor,W,padding=padding,strides=strides)  + b 
        #convolution = activation_fn(convolution)
	if batchnorm:
		convolution = tf_util.batch_norm_template(convolution, istraining, name, [0,1,2,3], bn_decay)
	return activation_fn( convolution )
	
def MaxPooling3D(inputtensor,dim1,dim2,dim3):	
	return tf_max_pool(inputtensor,dim1,dim2,dim3)

def AvgPooling3D(inputtensor,dim1,dim2,dim3):	
	return tf_avg_pool(inputtensor,dim1,dim2,dim3)

def AvgPooling3DStrides(inputtensor,dim1,dim2,dim3,dim1s,dim2s,dim3s):
	return tf_avg_pool_stride(inputtensor,dim1,dim2,dim3,dim1s,dim2s,dim3s)

def Flatten(inputtensor):
	return tf.contrib.layers.flatten(inputtensor)

def maxnorm(vector):
	return 1.0*vector/K.max(vector)
	
def maxnormshape(shape):
	return shape

def add(vector):
	s = K.sum(vector)
	return K.reshape(s,(1,1))

def lse(tensor):
	return tf.reshape ( (1.0/40.0)*tf.log(tf.reduce_mean(tf.exp(40*tensor))) , (1,1) )

def batch_norm(x, n_out, phase_train, scope='bn'):
	"""
	Batch normalization on convolutional maps.
	Args:
			x:           Tensor, 4D BHWD input maps
			n_out:       integer, depth of input maps
			phase_train: boolean tf.Varialbe, true indicates training phase
			scope:       string, variable scope
	Return:
			normed:      batch-normalized maps
	"""
	beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
															 name='beta', trainable=True)
	gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
																name='gamma', trainable=True)

	with tf.device('/cpu:0'):
		batch_mean, batch_var = tf.nn.moments(x, [-1], name='moments')
	ema = tf.train.ExponentialMovingAverage(decay=0.5)

	def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

	mean, var = tf.cond(phase_train,
								mean_var_with_update,
								lambda: (ema.average(batch_mean), ema.average(batch_var)))
	normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed

def batch_norm_wrapper(inputs, is_training, decay = 0.4):
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	if is_training is True:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2,3])
			train_mean = tf.assign(pop_mean,
														 pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var,
														pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
					return tf.nn.batch_normalization(inputs,
							batch_mean, batch_var, beta, scale, 1e-3)
	else:
			return tf.nn.batch_normalization(inputs,
					pop_mean, pop_var, beta, scale, 1e-3)
