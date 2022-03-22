from pycocotools.coco import COCO
import numpy as np
import random
import operator
import argparse
from functools import reduce
from collections import defaultdict
import os

def parse_args():
    parser = argparse.ArgumentParser(description='openset voc generator')
    parser.add_argument("--dir", default="datasets/voc_coco/ImageSets/Main", type=str, help="output dir")
    parser.add_argument("--ann_path", default="datasets/coco/annotations/instances_train2017.json", type=str, help="annotation path")
    return parser.parse_args()

def split_coco_trainval(ann_file, out_dir, min_sample_num=10, max_sample_num=50):
    image_dict = defaultdict(list)
    coco_instance = COCO(ann_file)
    for index, image_id in enumerate(coco_instance.imgToAnns):
        image_details = coco_instance.imgs[image_id]
        classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
        classes = set(classes)
        image_name = image_details['file_name'].split('.')[0]
        for cls in classes:
            image_dict[cls].append(image_name)

    for cls in image_dict.keys():
        image_dict[cls] = list(set(image_dict[cls]))

    image_train_dict = defaultdict(list)
    image_val_dict = defaultdict(list)
    num_arr = [len(image_dict[cls]) for cls in image_dict]
    min_num = min(num_arr)
    for cls in image_dict:
        image_dict_per_cls = image_dict[cls]
        num_to_sample = int(len(image_dict_per_cls) / min_num * min_sample_num)
        num_to_sample = min(num_to_sample, max_sample_num)
        random.shuffle(image_dict_per_cls)
        image_train_dict[cls].append(image_dict_per_cls[num_to_sample:])
        image_val_dict[cls].append(image_dict_per_cls[:num_to_sample])
    
    image_train_dict = reduce(operator.add, [x[0] for _,x in image_train_dict.items()])
    image_val_dict = reduce(operator.add, [x[0] for _,x in image_val_dict.items()])
    
    image_train_dict = set(image_train_dict)
    image_val_dict = set(image_val_dict)
    image_train_dict = [x for x in image_train_dict if x not in image_val_dict]


    with open(os.path.join(out_dir, "ImageSets/Main", "instances_train2017_train.txt"), "w") as f:
        f.writelines("\n".join(image_train_dict))
    with open(os.path.join(out_dir, "ImageSets/Main", "instances_train2017_val.txt"), "w") as f:
        f.writelines("\n".join(image_val_dict))

if __name__ == "__main__":
    args = parse_args()
    split_coco_trainval(args.ann_path, args.dir)