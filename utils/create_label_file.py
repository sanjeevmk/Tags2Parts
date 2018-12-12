import sys
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--shapenetSegRoot")
parser.add_argument("--shapeClass")
parser.add_argument("--partId")
parser.add_argument("--outputLabelFile")

args = parser.parse_args()

synsetMap = json.loads(open('../utils/synsetnames.json','r').read())
synsetId = synsetMap[args.shapeClass]

segFilesPath = os.path.join(args.shapenetSegRoot,synsetId,'expert_verified','points_label')

segFileNames = os.listdir(segFilesPath)

labelfHandle = open(args.outputLabelFile,'w')

for fname in segFileNames:
    if fname.endswith('.seg'):
        modelId = fname.split('.seg')[0]
        if args.partId in open(os.path.join(segFilesPath,fname),'r').read():
            label = '1'
        else:
            label = '0'
        labelfHandle.write(modelId+','+label+'\n')

labelfHandle.close()
