"""
Online demo at huggingface.
The link is: https://huggingface.co/spaces/csuhan/opendet2
"""
import os
os.system('pip install torch==1.9 torchvision')
os.system('pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html')
os.system('pip install timm opencv-python-headless')


import gradio as gr

from demo.predictor import VisualizationDemo
from detectron2.config import get_cfg
from opendet2 import add_opendet_config


model_cfgs = {
    "FR-CNN": ["configs/faster_rcnn_R_50_FPN_3x_baseline.yaml", "frcnn_r50.pth"],
    "OpenDet-R50": ["configs/faster_rcnn_R_50_FPN_3x_opendet.yaml", "opendet2_r50.pth"],
    "OpenDet-SwinT": ["configs/faster_rcnn_Swin_T_FPN_18e_opendet_voc.yaml", "opendet2_swint.pth"],
}


def setup_cfg(model):
    cfg = get_cfg()
    add_opendet_config(cfg)
    model_cfg = model_cfgs[model]
    cfg.merge_from_file(model_cfg[0])
    cfg.MODEL.WEIGHTS = model_cfg[1]
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.VIS_IOU_THRESH = 0.8
    cfg.freeze()
    return cfg


def inference(input, model):
    cfg = setup_cfg(model)
    demo = VisualizationDemo(cfg)
    # use PIL, to be consistent with evaluation
    predictions, visualized_output = demo.run_on_image(input)
    output = visualized_output.get_image()[:, :, ::-1]
    return output


iface = gr.Interface(
    inference,
    [
        "image",
        gr.inputs.Radio(
            ["FR-CNN", "OpenDet-R50", "OpenDet-SwinT"], default='OpenDet-R50'),
    ],
    "image")

iface.launch()
