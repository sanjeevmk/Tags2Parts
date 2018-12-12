from weak_train import *
from scipy.spatial import cKDTree as ckdt
from utils import binvox_rw
from sklearn.metrics import precision_recall_fscore_support as prfs

def getAsPc(grid,gtpc,scale,translate):
    dims = 64
    gtpts = gtpc[:,:3]
    bbmin = np.min(gtpts,axis=0)
    bbmax = np.max(gtpts,axis=0)
    center = 0.5*(bbmax-bbmin)
    w1s = np.where(grid==1.0)
    grid_xyz = [[x,y,z] for x,y,z in zip(w1s[0],w1s[1],w1s[2])]
    grid_xyz = np.array(grid_xyz)
    grid_xyz_sc = []
    for p in grid_xyz:
        trans_p = [0,0,0]
        trans_p[0] = scale* ((1/scale)*center[0]-0.5 + float((p[0]+0.5)/dims) )+ translate[0]
        trans_p[1] = scale* ((1/scale)*center[1]-0.5 + float((p[1]+0.5)/dims) )+ translate[1]
        trans_p[2] = scale* ((1/scale)*center[2]-0.5 + float((p[2]+0.5)/dims) )+ translate[2]
        grid_xyz_sc.append(trans_p)
    grid_xyz_sc = np.array(grid_xyz_sc)
    return grid_xyz_sc,grid_xyz

arch.load(args['sessionFile'])
#arch.initializer()
thresh = args['thresh']
if not os.path.exists(args['outputDir']):
    os.makedirs(args['outputDir'])

traingen = generator.voxelGeneratorPaths(trainInstances,args['batch_size'])
meanacc = 0.0 ; meanerr = 0.0
prs = [] ; recs = []
for j in range(len(trainInstances)):
    try:
        grids,paths = next(traingen)
        branch_1,branch_2,classVector = arch.sess.run(p2_branchouts + [phase3_class],feed_dict={vox_input:grids,is_training_pl:False})
        classPred = np.argmax(classVector,1)
        for bind in range(args['batch_size']):
            if classPred[bind]==1:
                continue

            voxPred = branch_1[bind]
            revVoxPred = voxPred[:,:,::-1,:]
            voxPred = np.maximum(revVoxPred,voxPred)

            voxPred[voxPred>=thresh] = 1
            voxPred[voxPred<thresh] = 0
            binvoxObj = binvox_rw.Voxels(voxPred,(voxdims,voxdims,voxdims),[0,0,0],1,'xzy')
            modelName = paths[bind].split('/')[-1]
            binvox_rw.write(binvoxObj,open(args['outputDir']+'/'+modelName+'_'+str(args['partId'])+'.binvox','wb'))
        if j%5==0:
            print('BATCH ' + str(j) + ' of ' + str(len(trainInstances)/args['batch_size']))
    except StopIteration:
        break
