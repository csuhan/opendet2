# Use Builtin Datasets

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

Detectron2 has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  VOC20{07,12}/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

The [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
contains configs and models that use these builtin datasets.

## Expected dataset structure for [COCO instance/keypoint detection](https://cocodataset.org/#download):

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

Some of the builtin tests (`dev/run_*_tests.sh`) uses a tiny version of the COCO dataset,
which you can download with `./datasets/prepare_for_tests.sh`.

## Expected dataset structure for [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html):
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```
