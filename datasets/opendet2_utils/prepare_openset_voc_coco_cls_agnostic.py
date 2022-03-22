# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import itertools
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
import argparse
from collections import defaultdict
import random
import operator
from functools import reduce

from detectron2.utils.file_io import PathManager


VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]

VOC_COCO_CLASS_NAMES = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES))

def parse_args():
    parser = argparse.ArgumentParser(description='openset voc generator')
    parser.add_argument("--dir", default="datasets/voc_coco", type=str, help="dataset dir")
    parser.add_argument("--in_split", default="instances_train2017", type=str, help="in split name")
    parser.add_argument("--out_split", default="instances_train2017_openset_cls_agn_5000", type=str, help="out split name")
    parser.add_argument("--start_class", default="20", type=int)
    parser.add_argument("--end_class", default="40", type=int)
    parser.add_argument("--post_num_sample", default="5000", type=int)
    return parser.parse_args()

def prepare_openset(dirname: str, in_split: str, out_split: str, start_class: int, end_class: int, post_num_sample_img: int):
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", in_split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))

    image_ids = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        classes = [obj.find("name").text for obj in tree.findall("object")]
        if set(classes).isdisjoint(VOC_COCO_CLASS_NAMES[:start_class]+VOC_COCO_CLASS_NAMES[end_class:]):
            image_ids.append(fileid)
    
    image_ids = set(image_ids)
    num_img = len(image_ids)
    # print(num_img)
    post_num_sample_img = min(num_img, post_num_sample_img)
    image_ids = random.sample(image_ids, post_num_sample_img)

    with open(os.path.join(dirname, "ImageSets", "Main", out_split + ".txt"), "w") as f:
        f.writelines("\n".join(image_ids)+"\n")

if __name__ == "__main__":
    args = parse_args()
    prepare_openset(args.dir, args.in_split, args.out_split, args.start_class, args.end_class, args.post_num_sample)