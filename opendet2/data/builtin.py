import os

from .voc_coco import register_voc_coco
from detectron2.data import MetadataCatalog


def register_all_voc_coco(root):
    SPLITS = [
        # VOC_COCO_openset
        ("voc_coco_20_40_test", "voc_coco", "voc_coco_20_40_test"),
        ("voc_coco_20_60_test", "voc_coco", "voc_coco_20_60_test"),
        ("voc_coco_20_80_test", "voc_coco", "voc_coco_20_80_test"),

        ("voc_coco_2500_test", "voc_coco", "voc_coco_2500_test"),
        ("voc_coco_5000_test", "voc_coco", "voc_coco_5000_test"),
        ("voc_coco_10000_test", "voc_coco", "voc_coco_10000_test"),
        ("voc_coco_20000_test", "voc_coco", "voc_coco_20000_test"),

        ("voc_coco_val", "voc_coco", "voc_coco_val"),

    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_voc_coco(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


if __name__.endswith(".builtin"):
    # Register them all under "./datasets"
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_voc_coco(_root)
