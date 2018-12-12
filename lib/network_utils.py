import tensorflow as tf

def glorot_fc_weight(name,inChannels,outChannels,seed=10):
	w = tf.get_variable( name,shape=[inChannels,outChannels],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,seed=seed) )
	b = tf.get_variable( name+'_b',shape=[outChannels],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,seed=seed) )
	return w,b

def glorot_conv2d(tensor,name,height,width,inChannels,outChannels,seed=10):
	w = tf.get_variable( name,shape=[height,width,inChannels,outChannels],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,seed=seed) )
	b = tf.get_variable( name+'_b',shape=[outChannels],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,seed=seed) )

	return tf.nn.relu(tf.nn.conv2d(tensor,w,strides=[1,1,1,1],padding='SAME') + b)

def avg_pool(tensor, height,width,name):
	return tf.nn.avg_pool(tensor, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding='SAME', name=name)

def max_pool(tensor, height,width, name):
	return tf.nn.max_pool(tensor, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding='SAME', name=name)

def getNetworkOutput(session,graphInput,graphOutput,inputSample):
	output = session.run([graphOutput], feed_dict={ graphInput: inputSample} )
	return output
