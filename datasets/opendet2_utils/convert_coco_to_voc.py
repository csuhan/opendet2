import xml.etree.cElementTree as ET
import os
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO

COCO2VOC_CLASS_NAMES = {
    "airplane": "aeroplane",
    "dining table": "diningtable",
    "motorcycle": "motorbike",
    "potted plant": "pottedplant",
    "couch": "sofa",
    "tv": "tvmonitor",
}

def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO to VOC style')
    parser.add_argument("--dir", default="datasets/voc_coco", type=str, help="dataset dir")
    parser.add_argument("--ann_path", default="datasets/coco/annotations/instances_train2017.json", type=str, help="annotation path")
    return parser.parse_args()

def convert_coco_to_voc(coco_annotation_file, target_folder):
    os.makedirs(os.path.join(target_folder, 'Annotations'), exist_ok=True)
    coco_instance = COCO(coco_annotation_file)
    image_ids = []
    for index, image_id in enumerate(tqdm(coco_instance.imgToAnns)):
        image_details = coco_instance.imgs[image_id]
        annotation_el = ET.Element('annotation')
        ET.SubElement(annotation_el, 'filename').text = image_details['file_name']

        size_el = ET.SubElement(annotation_el, 'size')
        ET.SubElement(size_el, 'width').text = str(image_details['width'])
        ET.SubElement(size_el, 'height').text = str(image_details['height'])
        ET.SubElement(size_el, 'depth').text = str(3)

        for annotation in coco_instance.imgToAnns[image_id]:
            object_el = ET.SubElement(annotation_el, 'object')
            cls_name = coco_instance.cats[annotation['category_id']]['name']
            if cls_name in COCO2VOC_CLASS_NAMES.keys():
                cls_name = COCO2VOC_CLASS_NAMES[cls_name]
            ET.SubElement(object_el,'name').text = cls_name
            # ET.SubElement(object_el, 'name').text = 'unknown'
            ET.SubElement(object_el, 'difficult').text = '0'
            bb_el = ET.SubElement(object_el, 'bndbox')
            ET.SubElement(bb_el, 'xmin').text = str(int(annotation['bbox'][0] + 1.0))
            ET.SubElement(bb_el, 'ymin').text = str(int(annotation['bbox'][1] + 1.0))
            ET.SubElement(bb_el, 'xmax').text = str(int(annotation['bbox'][0] + annotation['bbox'][2] + 1.0))
            ET.SubElement(bb_el, 'ymax').text = str(int(annotation['bbox'][1] + annotation['bbox'][3] + 1.0))
        
        file_name = image_details['file_name'].split('.')[0]
        image_ids.append(file_name)
        ET.ElementTree(annotation_el).write(os.path.join(target_folder, 'Annotations', file_name + '.xml'))

    imageset_dir = os.path.join(target_folder, 'ImageSets/Main')
    os.makedirs(imageset_dir, exist_ok=True)
    imageset_name = os.path.basename(coco_annotation_file).split(".json")[0] + ".txt"
    with open(os.path.join(imageset_dir, imageset_name), 'w')  as f:
        f.writelines("\n".join(image_ids)+'\n')
    

if __name__ == '__main__':
    args = parse_args()
    convert_coco_to_voc(args.ann_path, args.dir)
