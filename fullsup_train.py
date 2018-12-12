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
NB_EPOCH = 110
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
modelsPath = os.listdir(os.path.join(args['dataRoot'],synsetId))

allnames = open('./utils/Splits/All/' + args['shapeClass']+'.txt','r').read().splitlines()
testnames = open('./utils/Splits/Test/' + args['shapeClass']+'.txt','r').read().splitlines()
trainnames = [name for name in allnames if name not in testnames]
allModelInstances = []

allModelInstances = []
testInstances = []
for modelId in modelsPath:
    if (modelId not in trainnames) and (modelId not in testnames):
        continue
    fullPath = os.path.join(args['dataRoot'],synsetId,modelId)
    modelInstance = voxelgrid.VoxelGrid(fullPath)
    modelInstance.nParts = args['nparts']
    if modelId in trainnames:
        allModelInstances.append(modelInstance)
    else:
        testInstances.append(modelInstance)

trainInstances = allModelInstances[:int(0.8*len(allModelInstances))]
valInstances = allModelInstances[int(0.8*len(allModelInstances)):]

print(len(trainInstances),len(valInstances),len(testInstances))
vox_input = tf.placeholder(tf.float32, shape=(args['batch_size'],64,64,64,1))
is_training_pl = tf.placeholder(tf.bool, shape=())
_bn_decay = get_bn_decay(tf.Variable(0),args['batch_size'])
arch = Networks.Architectures()
base,nf = arch.wunet(vox_input,args['batch_size'],is_training_pl,_bn_decay)

branchouts = []
trueseg_holders = []
for i in range(args['nparts']):
    out  = layers.Convolution3DSig(base, nf, 3, 3, 3,1,init='glorot_uniform',name='branch_'+str(i+1),batchnorm=False,istraining=is_training_pl,bn_decay=_bn_decay)
    out = out*vox_input
    branchouts.append(out)
    seg_pl = tf.placeholder(tf.float32, shape=(args['batch_size'],64,64,64,1),name='part_'+str(i+1))
    trueseg_holders.append(seg_pl)

segloss = 0.0

for pid in range(args['nparts']):
    trueseg_part = trueseg_holders[pid]
    predseg_part = branchouts[pid]
    for i in range(args['batch_size']):
        segloss += Networks.cross_entropy(trueseg_part[i],predseg_part[i])
seglosses = {}
seglosses['segmentation_loss'] = segloss/(args['batch_size']*args['nparts'])

totalerror, trainer = arch.optimizer(seglosses)

arch.sess = tf.Session()

if __name__ == "__main__":
    arch.initializer()
    besterr = 1e9
    for i in range(NB_EPOCH):
        traingen = generator.voxelFullSegGenerator(trainInstances,args['batch_size'])
        meanerr = 0.0
        for j in range(len(trainInstances)):
            try:
                grids,segs = next(traingen)
                fd = {vox_input:grids,is_training_pl:True}
                for pid in range(args['nparts']):
                    fd[trueseg_holders[pid]] = segs[:,pid,:,:,:,:]

                _,error = arch.sess.run([trainer,totalerror],feed_dict=fd)
                meanerr += error
                if j%5==0:
                    print(' Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(trainInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1))
            except StopIteration:
                break

        valgen = generator.voxelFullSegGenerator(valInstances,args['batch_size'])
        meanerr = 0.0
        for j in range(len(valInstances)):
            try:
                grids,segs = next(valgen)
                fd = {vox_input:grids,is_training_pl:False}
                for pid in range(args['nparts']):
                    fd[trueseg_holders[pid]] = segs[:,pid,:,:,:,:]
                error = arch.sess.run([totalerror],feed_dict=fd)[0]
                meanerr += error
                if j%5==0:
                    print('Validation Epoch ' +str(i) + ' Batch ' +str(j) + ' of ' + str(len(valInstances)/args['batch_size']) + ' - Loss: ',meanerr/(j+1))
            except StopIteration:
                break

        valerror = meanerr/j
        if valerror < besterr:
            besterr = valerror
            tfsaver = tf.train.Saver(max_to_keep=None)
            tfsaver.save(arch.sess,args['sessionFile'])
