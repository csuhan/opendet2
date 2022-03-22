#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (default_argument_parser, default_setup, hooks,
                               launch)
from detectron2.evaluation import verify_results
from detectron2.utils.logger import setup_logger
from opendet2 import OpenDetTrainer, add_opendet_config, builtin


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add opendet config
    add_opendet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Note: we use the key ROI_HEAD.NUM_KNOWN_CLASSES
    # for open-set data processing and evaluation.
    if 'RetinaNet' in cfg.MODEL.META_ARCHITECTURE:
        cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = cfg.MODEL.RETINANET.NUM_KNOWN_CLASSES
    # add output dir if not exist
    if cfg.OUTPUT_DIR == "./output":
        config_name = os.path.basename(args.config_file).split(".yaml")[0]
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, config_name)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR,
                 distributed_rank=comm.get_rank(), name="opendet2")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = OpenDetTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = OpenDetTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(OpenDetTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = OpenDetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
