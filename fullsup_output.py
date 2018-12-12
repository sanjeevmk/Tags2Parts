from fullsup_train import *
from scipy.spatial import cKDTree as ckdt
from utils import binvox_rw
from sklearn.metrics import precision_recall_fscore_support as prfs

arch.load(args['sessionFile'])
thresh = args['thresh']
if not os.path.exists(args['outputDir']):
    os.makedirs(args['outputDir'])

testgen = generator.voxelGeneratorPaths(testInstances,args['batch_size'])
meanacc = 0.0 ; meanerr = 0.0
prs = [] ; recs = []
for j in range(len(testInstances)):
    try:
        grids,paths = next(testgen)
        branches = arch.sess.run(branchouts,feed_dict={vox_input:grids,is_training_pl:False})
        for bind in range(args['batch_size']):
            for pid in range(args['nparts']):
                voxPred = branches[pid][bind]
                voxPred[voxPred>=thresh] = 1
                voxPred[voxPred<thresh] = 0
                binvoxObj = binvox_rw.Voxels(voxPred,(voxdims,voxdims,voxdims),[0,0,0],1,'xzy')
                modelName = paths[bind].split('/')[-1]
                binvox_rw.write(binvoxObj,open(args['outputDir']+'/'+modelName+'_'+str(pid+1)+'.binvox','wb'))
        if j%5==0:
            print('BATCH ' + str(j) + ' of ' + str(len(testInstances)/args['batch_size']))
    except StopIteration:
        break
