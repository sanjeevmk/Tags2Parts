import os
import random
import numpy as np
random.seed(2016)
import csv
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import binvox_rw

dims = 64 

class VoxelGrid:
    def __init__(self,binvoxPath):
        self.binvoxPath = binvoxPath
        self.labelvector = None
        self.cache = None

    def readGrid(self,return_params=False):
        if self.cache is None:
            binvoxPath = self.binvoxPath + '/model.binvox'

            with open(binvoxPath,'rb') as f:
                binvoxObj = binvox_rw.read_as_3d_array(f)
                scale = binvoxObj.scale
                translate = binvoxObj.translate
                self.scale = scale
                self.translate = translate
                voxelgrid = np.reshape(binvoxObj.data,(dims,dims,dims,1))
                voxelgrid = voxelgrid.astype('float32')
            self.cache = voxelgrid
        else:
            voxelgrid = self.cache
            scale = self.scale
            translate = self.translate

        if not return_params:
            return voxelgrid
        else:
            return voxelgrid,scale,translate

    def readGridAndSeg(self,return_params=False):
        binvoxPath = self.binvoxPath + '/model.binvox'

        with open(binvoxPath,'rb') as f:
            binvoxObj = binvox_rw.read_as_3d_array(f)
            scale = binvoxObj.scale
            translate = binvoxObj.translate
            self.scale = scale
            self.translate = translate
            voxelgrid = np.reshape(binvoxObj.data,(dims,dims,dims,1))
            voxelgrid = voxelgrid.astype('float32')

        pointCloud = np.array(list(csv.reader(open(self.ptsPath,'r'),delimiter=' ')))
        strSeg = open(self.segPath).read().splitlines()
        segLabels = [int(l) for l in strSeg]
        if not return_params:
            return voxelgrid,pointCloud,segLabels
        else:
            return voxelgrid,scale,translate,pointCloud,segLabels

    def readGridFullSeg(self):
        binvoxPath = self.binvoxPath + '/model.binvox'

        with open(binvoxPath,'rb') as f:
            binvoxObj = binvox_rw.read_as_3d_array(f)
            voxelgrid = np.reshape(binvoxObj.data,(dims,dims,dims,1))
            voxelgrid = voxelgrid.astype('float32')

        partSegs = []
        for i in range(1,self.nParts+1):
            segPath = self.binvoxPath + '/model_'+str(i)+'.binvox'
            with open(segPath,'rb') as f:
                segBinvoxObj = binvox_rw.read_as_3d_array(f)
                segGrid = np.reshape(segBinvoxObj.data,(dims,dims,dims,1))
                segGrid = segGrid.astype('float32')
                partSegs.append(segGrid)

        return voxelgrid,partSegs
