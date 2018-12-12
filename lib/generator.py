import random
import numpy as np
def voxelGeneratorPaths(instances,batch_size):
    grids = [] ; paths = []
    random.shuffle(instances)

    for modelIns in instances:
        grid = modelIns.readGrid()
        grids.append(grid) ; paths.append(modelIns.binvoxPath)

        if len(grids) == batch_size:
            yield np.array(grids),paths
            grids = []
            paths = []

    if len(grids)>0:
        for i in range(len(grids),batch_size):
            grids.append(grids[-1])
            paths.append(paths[-1])
        yield grids,paths
        grids = []
        paths = []

def voxelGenerator(instances,batch_size):
    grids = [] ; labels = []
    random.shuffle(instances)

    for modelIns in instances:
        grid = modelIns.readGrid()
        lvector = modelIns.labelvector
        grids.append(grid) ; labels.append(lvector)

        if len(grids) == batch_size:
            yield np.array(grids),np.array(labels)
            grids = []
            labels = []

    if len(grids)>0:
        for i in range(len(grids),batch_size):
            grids.append(grids[-1])
            labels.append(labels[-1])
        yield grids,labels
        grids = []
        labels = []

def voxelAndSegGenerator(instances,batch_size):
    grids = [] ; labels = [] ; scales = [] ; translates = [] ; clouds = [] ; segs = []
    random.shuffle(instances)

    for modelIns in instances:
        grid,scale,translate,cloud,seg = modelIns.readGridAndSeg(return_params=True)
        lvector = modelIns.labelvector
        grids.append(grid) ; labels.append(lvector)
        scales.append(scale) ; translates.append(translate)
        clouds.append(cloud[:,:3]) ; segs.append(seg)

        if len(grids) == batch_size:
            yield np.array(grids),scales,translates,clouds,segs,np.array(labels)
            grids = []
            labels = []
            scales = []
            translates = []
            clouds = []
            segs = []

    if len(grids)>0:
        for i in range(len(grids),batch_size):
            grids.append(grids[-1])
            labels.append(labels[-1])
            scales.append(scales[-1])
            translates.append(translates[-1])
            clouds.append(clouds[-1])
            segs.append(segs[-1])
        yield grids,scales,translates,clouds,segs,labels
        grids = []
        labels = []
        scales = []
        translates = []
        clouds = []
        segs = []

def voxelFullSegGenerator(instances,batch_size):
    grids = [] ; segs = []
    random.shuffle(instances)

    for modelIns in instances:
        grid,seg = modelIns.readGridFullSeg()
        grids.append(grid) 
        segs.append(seg)

        if len(grids) == batch_size:
            yield np.array(grids),np.array(segs)
            grids = []
            segs = []

    if len(grids)>0:
        for i in range(len(grids),batch_size):
            grids.append(grids[-1])
            segs.append(segs[-1])
        yield np.array(grids),np.array(segs)
        grids = []
        segs = []
