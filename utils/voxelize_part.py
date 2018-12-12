import sys
import os
import argparse
import json
import binvox_rw
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knc

def getVoxelGrid(binvoxPath):
    binvoxObj = binvox_rw.read_as_3d_array(open(binvoxPath,'rb'))
    scale = binvoxObj.scale
    translate = binvoxObj.translate
    voxelgrid = binvoxObj.data.astype('float32')
    dims = binvoxObj.dims

    return voxelgrid,dims,scale,translate

def getPointCloud(ptsPath):
    pointList = list(csv.reader(open(ptsPath,'r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC))
    return pointList

def getLabelList(segPath):
    segLabels = open(segPath,'r').read().splitlines()
    segLabels = [int(l) for l in segLabels]
    return segLabels

def voxelizePart(grid,scale,translate,dims,cloud,labels,partId,outputPath):
    bbmin = np.min(cloud,axis=0)
    bbmax = np.max(cloud,axis=0)
    center = 0.5*(bbmax-bbmin)
    w1s = np.where(grid==1)
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
    #grid_xyz_sc is now in the same coordinate frame as the point-cloud

    clf = knc(n_neighbors=1)
    clf.fit(cloud,labels)
    voxelLabels = clf.predict(grid_xyz_sc)
    partIndices = voxelLabels==partId
    partVoxelIndices = grid_xyz[partIndices,:]
    partvox = np.zeros((dims,dims,dims,1))
    partvox[partVoxelIndices[:,0],partVoxelIndices[:,2],partVoxelIndices[:,1],0] = 1
    partvox = partvox.astype('int')
    partbinvox = binvox_rw.Voxels(partvox,(dims,dims,dims),[0,0,0],1,'xzy')
    partname = 'model_'+str(partId)+'.binvox'
    binvox_rw.write(partbinvox,open(os.path.join(outputPath,partname),'wb'))

def binarizeLabelsByPart(labelList,partId):
    labelList[labelList!=partId] = 0
    labelList[labelList==partId] = 1
    return labelList

parser = argparse.ArgumentParser()
parser.add_argument("--shapenetModelRoot")
parser.add_argument("--shapenetSegRoot")
parser.add_argument("--shapeClass")
parser.add_argument("--partId")

args = parser.parse_args()

synsetMap = json.loads(open('synsetnames.json','r').read())
synsetId = synsetMap[args.shapeClass]

segFilesPath = os.path.join(args.shapenetSegRoot,synsetId,'expert_verified','points_label')
ptsFilesPath = os.path.join(args.shapenetSegRoot,synsetId,'points')
modelFilesPath = os.path.join(args.shapenetModelRoot,synsetId)

segFileNames = os.listdir(segFilesPath)


for fname in segFileNames:
    if fname.endswith('.seg'):
        modelId = fname.split('.seg')[0]
        modelFileName = os.path.join(modelFilesPath,modelId,'model.binvox')
        pointsFileName = os.path.join(ptsFilesPath,modelId+'.pts')
        labelsFileName = os.path.join(segFilesPath,fname)
        grid,dims,scale,translate = getVoxelGrid(modelFileName)
        pointList = getPointCloud(pointsFileName)
        segLabels = getLabelList(labelsFileName)
        voxelizePart(grid,scale,translate,dims[0],np.array(pointList),np.array(segLabels),int(args.partId),os.path.join(modelFilesPath,modelId))
        #partBinaryLabels = binarizeLabelsByPart(np.array(segLabels),int(args.partId))
        #partBinaryLabelF = os.path.join(modelFilesPath,modelId)+'/part_'+args.partId+'.seg'
        #csv.writer(open(partBinaryLabelF,'w'),delimiter='\n').writerows(np.expand_dims(partBinaryLabels,-1))
