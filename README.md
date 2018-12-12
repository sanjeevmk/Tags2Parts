## Tags2Parts : Discovering Semantic Regions from Shape Tags
### This is the source repository of Tags2Parts as detailed in : https://arxiv.org/abs/1708.06673

#### Data
The Shapenet part annotation data is available [here](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip)

The corresponding Shapenet 3D models - **ShapeNetCore.v1** can be downloaded from the [Shapenet website](https://www.shapenet.org/). 

#### Pre-Processing
- ##### Voxelize models
Writes `model.binvox` alongside `model.obj` files inside the ShapeNet folder.
```
cd utils
bash voxelize.sh /path/to/ShapeNetCore.v1/
```
- ##### Create the label files
The label files used for the 6 parts are provided under the `labels/` folder. For different parts or annotation data however, follow steps as outlined below.
Create the Has-Part/Lacks-Part classification-label files for each of the 6 parts. Outputs a label file for each part in CSV format, with 2 columns as `modelId,[1,0]` where `1` indicates that the model has the said part, while `0` indicates lack of it.
```
cd utils
python create_label_file.py --shapenetSegRoot /path/to/PartAnnotation/ --shapeClass [shapeClass like Airplane] --partId [1,2,3,4,5,6] --outputLabelFile /output/label/file
```
- ##### Only for fully supervised segmentation
Create voxelized ground-truth segmentation data, for fully supervised training.
```
cd utils
python voxelize_part.py --shapenetModelRoot /path/to/ShapeNetCore.v1/ --shapenetSegRoot /path/to/PartAnnotation --shapeClass [Airplane,Car,Chair etc] --partId [1,2,3,4,5,6]
```
Valid `shapeClass` names are any of the LHS names in `utils/synsetnames.json`, with the exception of `ChairBack,Ship,Bed` (the latter are only used for weakly supervised segmentation)
Valid `partId` varies from 1 to 6 depending on the `shapeClass`. Refer paper for number of parts associated with each `shapeClass`. Eg, if 4 parts, then `partId` can be any of 1,2,3,4. For a given `shapeClass` repeat above for each `partId`. 

Creates `model_<partId>.binvox` for each `partId` alongside corresponding `model.obj` files in the ShapeNet data root (ShapeNetCore.v1).

#### Training
- ##### Weakly supervised
`python weak_train.py weak_config.json`
Arguments are passed via `weak_config.json`.
It has the following parameters:
```
synsetMap : Maps shapeClass names to synsetIds. The map file to use is provided under utils/ as synsetnames.json.
shapeClass : One of Airplane,Chair,Car,ChairBack,Ship,Bed.
labelFile : The classification label file for the corresponding part. One of the 6 files under the labels/ folder, or any label file as created before in the pre-processing section.
dataRoot : Full path to ShapeNetCore.v1 root.
segRoot : Full path to Part Annotation root.
batch_size : 5 , by default.
partId : (For the 6 supported parts) 4 for Propeller or Arm, 1 for Roof, Chair Back, Sail and Bed Head. (For any additional parts) use partId using which the label file was generated.
sessionFile : Path to save trained model.
outputDir : To be used during prediction ; path to save precision/recall plot file and the segmented binvox files.
plotFile : Name of the plot file; this file will be created under outputDir.
thresh : The threshold, between 0 and 1, to be applied on the predicted segments output by the network. The binarized segment will be saved as modelId_partId.binvox under outputDir.
```
- ##### Fully supervised
`python fullsup_train.py fullsup_config.json`
Arguments are passed via `fullsup_config.json` which is a similar file as `weak_config.json` with fewer parameters, and the addition of `npart` which specifies the number of segmented parts; to be set according to `shapeClass`. 
`shapeClass` can be any of the LHS names in `utils/sysnsetnames.json` , with the exception of `ChairBack,Ship,Bed`.

- #### Predicting
- ##### Weakly supervised
`python weak_eval.py weak_config.json`
`weak_config.json` same as that used for training. Outputs a plot file with the name `plotFile` under `outputDir`. `plotFile` output is a CSV file with 2 columns as `Recall,Precision`.

`python weak_output.py weak_config.json`
Outputs the segmented parts as `model_<partId>.binvox`, under `outputDir`. 
`utils/viewvox /path/to/binvoxFile` can be used to view the generated segment.

- ##### Fully supervised
`python fullsup_output.py fullsup_config.json`
Config file to be same as that used for training. Outputs `model_<partId>.binvox` under `outputDir`, for `partId` in `<1,2...nparts>`.
`utils/viewvox /path/to/binvoxFile` can be used to view the generated segment.
