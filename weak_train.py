import numpy as np
np.random.seed(2016)
import random
random.seed(2016)
import os
import csv
import json
import sys
from lib import voxelgrid,generator
import tensorflow as tf
from lib import Networks
from lib import layers
tf.set_random_seed(2016)

voxdims = 64
NB_EPOCH = 100
P2_NB_EPOCH = 50
P3_NB_EPOCH = 10

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

args = json.loads(open(sys.argv[1],'r').read())

synsetMap = json.loads(open(args['synsetMap'],'r').read())
synsetId = synsetMap[args['shapeClass']]
labelFile = args['labelFile']

segRoot = args['segRoot']
segDirPath = os.path.join(segRoot,synsetId)
labelData = list(csv.reader(open(labelFile,'r'),delimiter=','))

labelMap = {}
for ld in labelData:
    labelMap[ld[0]] = int(ld[1])

modelsPath = os.listdir(os.path.join(args['dataRoot'],synsetId))

allModelInstances = []
partPositive = 0
partNegative = 0

for modelId in modelsPath:
    if modelId not in labelMap:
        continue

    fullPath = os.path.join(args['dataRoot'],synsetId,modelId)
    modelInstance = voxelgrid.VoxelGrid(fullPath)
    lvector = [0,0]
    if labelMap[modelId] == 1:
        partPositive += 1
    else:
        partNegative += 1
    lvector[1-labelMap[modelId]] = 1
    modelInstance.labelvector = lvector
    segFilePath = os.path.join(segDirPath,'expert_verified','points_label') + '/' + modelId + '.seg'
    ptsFilePath = os.path.join(segDirPath,'points') + '/' + modelId + '.pts'
    modelInstance.segPath = segFilePath
    modelInstance.ptsPath = ptsFilePath
    allModelInstances.append(modelInstance)



trainInstances = allModelInstances[:int(0.8*len(allModelInstances))]
valInstances = allModelInstances[int(0.8*len(allModelInstances)):]

print(len(trainInstances),len(valInstances))
vox_input = tf.placeholder(tf.float32, shape=(args['batch_size'],64,64,64,1))
is_training_pl = tf.placeholder(tf.bool, shape=())
labels_pl = tf.placeholder(tf.float32, shape=(args['batch_size'],2))
_bn_decay = get_bn_decay(tf.Variable(0),args['batch_size'])
arch = Networks.Architectures()
base,nf = arch.wunet(vox_input,args['batch_size'],is_training_pl,_bn_decay)

#PHASE 1 network
base_globalpooled = layers.MaxPooling3D(base,64,64,64)
base_gp_flat = layers.Flatten(base_globalpooled)
phase1_class = tf.nn.softmax(layers.DenseRelu(base_gp_flat,nf,2,'glorot_uniform','outdense',istraining=is_training_pl,bn_decay=_bn_decay))

#PHASE 1 loss
phase1_class_loss = 0
phase1_loss = {}
monitor = {}
for i in range(args['batch_size']):
    y_pred_f = tf.gather(phase1_class,i)
    y_true_f = tf.gather(labels_pl,i)

    ce = -tf.reduce_mean( (  (y_true_f*tf.log(y_pred_f+ 1e-9)) + ((1-y_true_f) * tf.log(1 - y_pred_f+ 1e-9)) )  , name='p1_xentropy' )
    phase1_class_loss+=ce

phase1_loss['cross_entropy'] = phase1_class_loss/args['batch_size']
#phase1_loss['cross_entropy'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=phase1_class,labels=labels_pl))
p1_correct = tf.equal(tf.argmax(labels_pl,1),tf.argmax(phase1_class,1))
p1_totalerror, p1_trainer = arch.optimizer(phase1_loss)
monitor['p1_accuracy'] = tf.reduce_mean(tf.cast(p1_correct,tf.float32))

#PHASE 2 network
p2_branchouts = []
for i in range(2):
    out  = layers.Convolution3DSig(base, nf, 3, 3, 3,1,init='glorot_uniform',name='p2_'+str(i+1),batchnorm=False,istraining=is_training_pl,bn_decay=_bn_decay)
    out = out*vox_input
    p2_branchouts.append(out)

p2outTensors = []
for i in range(2):
    branchPred = layers.MaxPooling3D(p2_branchouts[i],64,64,64)
    p2outTensors.append(branchPred)

phase2_class = tf.squeeze(tf.concat(p2outTensors,1))
#PHASE 2 loss
phase2_class_loss = 0
phase2_loss = {}
for i in range(args['batch_size']):
    y_pred_f = tf.gather(phase2_class,i)
    y_true_f = tf.gather(labels_pl,i)

    ce = -tf.reduce_mean( (  (y_true_f*tf.log(y_pred_f+ 1e-9)) + ((1-y_true_f) * tf.log(1 - y_pred_f+ 1e-9)) )  , name='p2_xentropy' )
    phase2_class_loss+=ce
phase2_loss['cross_entropy'] = phase2_class_loss/args['batch_size']
#phase2_loss['cross_entropy'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=phase2_class,labels=labels_pl))
p2_correct = tf.equal(tf.argmax(labels_pl,1),tf.argmax(phase2_class,1))
p2_totalerror, p2_trainer = arch.optimizer(phase2_loss)
monitor['p2_accuracy'] = tf.reduce_mean(tf.cast(p2_correct,tf.float32))

#PHASE 3 network
p3outTensors = []
for i in range(2):
    branchPred = layers.AvgPooling3D(p2_branchouts[i],2,2,2)
    branchPred = layers.MaxPooling3D(branchPred,32,32,32)
    p3outTensors.append(branchPred)

phase3_class = tf.squeeze(tf.concat(p3outTensors,1))

#PHASE 3 loss
phase3_class_loss = 0
phase3_loss = {}
for i in range(args['batch_size']):
    y_pred_f = tf.gather(phase3_class,i)
    y_true_f = tf.gather(labels_pl,i)

    ce = -tf.reduce_mean( (  (y_true_f*tf.log(y_pred_f+ 1e-9)) + ((1-y_true_f) * tf.log(1 - y_pred_f+ 1e-9)) )  , name='p1_xentropy' )
    phase3_class_loss+=ce
phase3_loss['cross_entropy'] = phase3_class_loss/args['batch_size']
#phase3_loss['cross_entropy'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=phase3_class,labels=labels_pl))
p3_correct = tf.equal(tf.argmax(labels_pl,1),tf.argmax(phase3_class,1))
p3_totalerror, p3_trainer = arch.optimizer(phase3_loss)
monitor['p3_accuracy'] = tf.reduce_mean(tf.cast(p3_correct,tf.float32))
arch.sess = tf.Session()

if __name__ == "__main__":
    arch.initializer()
    bestacc = -1
    for i in range(NB_EPOCH):
        traingen = generator.voxelGenerator(trainInstances,args['batch_size'])
        meanacc = 0.0 ; meanerr = 0.0
        for j in range(len(trainInstances)):
            try:
                grids,labels = next(traingen)
                _,error,acc = arch.sess.run([p1_trainer,p1_totalerror,monitor['p1_accuracy']],feed_dict={vox_input:grids,labels_pl:labels,is_training_pl:True})
                meanerr += error
                meanacc += acc
                if j%5==0:
                    print('Phase 1 Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(trainInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1),'Accuracy: ',meanacc/(j+1))
            except StopIteration:
                break

        tracc = round(meanacc/j,2)
        valgen = generator.voxelGenerator(valInstances,args['batch_size'])
        meanacc = 0.0 ; meanerr = 0.0
        for j in range(len(valInstances)):
            try:
                grids,labels = next(valgen)
                error,acc = arch.sess.run([p1_totalerror,monitor['p1_accuracy']],feed_dict={vox_input:grids,labels_pl:labels,is_training_pl:False})
                meanerr += error
                meanacc += acc
                if j%5==0:
                    print('Phase 1 Validation Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(valInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1),'Accuracy: ',meanacc/(j+1))
            except StopIteration:
                break

        valacc = round(meanacc/j,2)
        if valacc > bestacc:
            bestacc = valacc
            tfsaver = tf.train.Saver(max_to_keep=None)
            tfsaver.save(arch.sess,args['sessionFile'])
        
        if tracc >= 0.95 and valacc >= 0.95:
            break

    #arch.load(args['sessionFile'])
    #arch.initializer()
    bestacc = -1
    for i in range(P2_NB_EPOCH):
        traingen = generator.voxelGenerator(trainInstances,args['batch_size'])
        meanacc = 0.0 ; meanerr = 0.0
        for j in range(len(trainInstances)):
            try:
                grids,labels = next(traingen)
                _,error,acc = arch.sess.run([p2_trainer,p2_totalerror,monitor['p2_accuracy']],feed_dict={vox_input:grids,labels_pl:labels,is_training_pl:True})
                meanerr += error
                meanacc += acc
                if j%5==0:
                    print('Phase 2 Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(trainInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1),'Accuracy: ',meanacc/(j+1))
            except StopIteration:
                break

        tracc = round(meanacc/j,2)
        valgen = generator.voxelGenerator(valInstances,args['batch_size'])
        meanacc = 0.0 ; meanerr = 0.0
        for j in range(len(valInstances)):
            try:
                grids,labels = next(valgen)
                error,acc = arch.sess.run([p2_totalerror,monitor['p2_accuracy']],feed_dict={vox_input:grids,labels_pl:labels,is_training_pl:False})
                meanerr += error
                meanacc += acc
                if j%5==0:
                    print('Phase 2 Validation Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(valInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1),'Accuracy: ',meanacc/(j+1))
            except StopIteration:
                break

        valacc = round(meanacc/j,2)
        
        if valacc > bestacc:
            bestacc = valacc
            tfsaver = tf.train.Saver(max_to_keep=None)
            tfsaver.save(arch.sess,args['sessionFile'])
        if tracc >= 0.97 and valacc >= 0.97:
            break
    #arch.load(args['sessionFile'])
    #arch.initializer()

    bestacc = -1
    for i in range(P3_NB_EPOCH):
        traingen = generator.voxelGenerator(trainInstances,args['batch_size'])
        meanacc = 0.0 ; meanerr = 0.0
        for j in range(len(trainInstances)):
            try:
                grids,labels = next(traingen)
                _,error,acc = arch.sess.run([p3_trainer,p3_totalerror,monitor['p3_accuracy']],feed_dict={vox_input:grids,labels_pl:labels,is_training_pl:True})
                meanerr += error
                meanacc += acc
                if j%5==0:
                    print('Phase 3 Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(trainInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1),'Accuracy: ',meanacc/(j+1))
            except StopIteration:
                break

        tracc = round(meanacc/j,2)
        valgen = generator.voxelGenerator(valInstances,args['batch_size'])
        meanacc = 0.0 ; meanerr = 0.0
        for j in range(len(valInstances)):
            try:
                grids,labels = next(valgen)
                error,acc = arch.sess.run([p3_totalerror,monitor['p3_accuracy']],feed_dict={vox_input:grids,labels_pl:labels,is_training_pl:False})
                meanerr += error
                meanacc += acc
                if j%5==0:
                    print('Phase 3 Validation Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(valInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1),'Accuracy: ',meanacc/(j+1))
            except StopIteration:
                break

        valacc = round(meanacc/j,2)
        
        if valacc > bestacc:
            bestacc = valacc
            tfsaver = tf.train.Saver(max_to_keep=None)
            tfsaver.save(arch.sess,args['sessionFile'])
