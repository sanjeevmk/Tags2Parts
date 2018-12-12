from weak_train import *
from scipy.spatial import cKDTree as ckdt
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

trainInstances = trainInstances 
traingen = generator.voxelAndSegGenerator(trainInstances,args['batch_size'])
meanacc = 0.0 ; meanerr = 0.0
prs = [] ; recs = []
for j in range(len(trainInstances)):
    try:
        grids,scales,translates,clouds,segs,labels = next(traingen)
        branch_1,branch_2,classVector,acc = arch.sess.run(p2_branchouts + [phase3_class] + [monitor['p3_accuracy']],feed_dict={vox_input:grids,labels_pl:labels,is_training_pl:False})
        classPred = np.argmax(classVector,1)
        truePred = np.argmax(labels,1)
        for bind in range(args['batch_size']):
            if classPred[bind]==1 and truePred[bind]==1:
                '''
                x = np.arange(0.0,1.01,0.01)
                thresh_pr = [1]*len(x)
                thresh_rec = [1]*len(x)
                prs.append(thresh_pr)
                recs.append(thresh_rec)
                '''
                continue
            #if classPred[bind]==1 and truePred[bind]==0:
            if classPred[bind] != truePred[bind]:
                x = np.arange(0.0,1.01,0.01)
                thresh_pr = [0]*len(x)
                thresh_rec = [0]*len(x)
                prs.append(thresh_pr)
                recs.append(thresh_rec)
                continue

            cloud = clouds[bind].astype('float32')
            seg = np.array(segs[bind])
            if seg.shape[0]==0:
                seg = np.zeros((cloud.shape[0],))
            grid = grids[bind]
            scale = scales[bind]
            translate = translates[bind]
            seg[seg!=args['partId']] = 0
            seg[seg==args['partId']] = 1
            grid_xyz_pc,grid_xyz = getAsPc(grid,cloud,scale,translate)

            if classPred[bind]==0:
                voxPred = branch_1[bind]
            else:
                voxPred = branch_2[bind]
            revVoxPred = voxPred[:,:,::-1,:]
            voxPred = np.maximum(revVoxPred,voxPred)
            voxPredLabels = voxPred[grid_xyz[:,0],grid_xyz[:,1],grid_xyz[:,2],0]
            tree = ckdt(grid_xyz_pc)
            _,inds = tree.query(cloud,k=1)
            pcPred = voxPredLabels[inds]
            thresh_pr = [] ; thresh_re = []
            for thresh in np.arange(0.0,1.01,0.01):
                threshPred = np.array(pcPred)
                threshPred[threshPred >= thresh] = 1
                threshPred[threshPred < thresh] = 0
                p,r,f,s = prfs(seg.flatten(),threshPred.flatten(),average='binary')
                thresh_pr.append(p)
                thresh_re.append(r)
            prs.append(thresh_pr)
            recs.append(thresh_re)

        meanacc += acc
        if j%5==0:
            print('BATCH ' + str(j) + ' of ' + str(len(trainInstances)/args['batch_size']),' PREDICTION ACCURACY: ',meanacc/(j+1))
    except StopIteration:
        if not os.path.exists(args['outputDir']):
            os.makedirs(args['outputDir'])
        prs = np.array(prs)
        recs = np.array(recs)
        pr_mean = np.mean(prs,axis=0)
        rec_mean = np.mean(recs,axis=0)
        pr_mean = np.expand_dims(pr_mean,-1)
        rec_mean = np.expand_dims(rec_mean,-1)
        output = np.hstack([rec_mean[::-1],pr_mean[::-1]])
        output = np.vstack([['Recall','Precision'],output])
        csv.writer(open(os.path.join(args['outputDir'],args['plotFile']),'w'),delimiter=',').writerows(output)
        break
