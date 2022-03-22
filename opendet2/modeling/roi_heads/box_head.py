# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.roi_heads import ROI_BOX_HEAD_REGISTRY
from detectron2.utils.registry import Registry
from torch import nn


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNSeparateConvFCHead(nn.Module):
    """
    FastRCNN with separate ConvFC layers
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims

        self._output_size = (input_shape.channels,
                             input_shape.height, input_shape.width)

        self.reg_conv_norm_relus = self._add_conv_norm_relus(
            self._output_size[0], conv_dims, conv_norm)
        self.cls_conv_norm_relus = self._add_conv_norm_relus(
            self._output_size[0], conv_dims, conv_norm)
        conv_dim = self._output_size[0] if len(conv_dims) == 0 else conv_dims[-1]
        self._output_size = (
            conv_dim, self._output_size[1], self._output_size[2])

        self.reg_fcs = self._add_fcs(np.prod(self._output_size), fc_dims)
        self.cls_fcs = self._add_fcs(np.prod(self._output_size), fc_dims)
        self._output_size = self._output_size if len(fc_dims)==0 else fc_dims[-1]

        for layer in self.reg_conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.cls_conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.cls_fcs:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
        for layer in self.reg_fcs:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }

    def _add_conv_norm_relus(self, input_dim, conv_dims, conv_norm):
        conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                input_dim,
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            input_dim = conv_dim
            conv_norm_relus.append(conv)

        return nn.Sequential(*conv_norm_relus)

    def _add_fcs(self, input_dim, fc_dims):
        fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                fcs.append(nn.Flatten())
            fc = nn.Linear(int(input_dim), fc_dim)
            fcs.append(fc)
            fcs.append(nn.ReLU())
            input_dim = fc_dim
        return nn.Sequential(*fcs)

    def forward(self, x):
        reg_feat = x
        cls_feat = x
        if len(self.conv_dims) > 0:
            reg_feat = self.reg_conv_norm_relus(x)
            cls_feat = self.cls_conv_norm_relus(x)
        if len(self.fc_dims) > 0:
            reg_feat = self.reg_fcs(reg_feat)
            cls_feat = self.cls_fcs(cls_feat)
        return reg_feat, cls_feat

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNSeparateDropoutConvFCHead(nn.Module):
    """Add dropout before each conv/fc layer
    """
    def _add_conv_norm_relus(self, input_dim, conv_dims, conv_norm):
        conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                input_dim,
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            input_dim = conv_dim
            conv_norm_relus.append(nn.Dropout2d(p=0.5))
            conv_norm_relus.append(conv)

        return nn.Sequential(*conv_norm_relus)

    def _add_fcs(self, input_dim, fc_dims):
        fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                fcs.append(nn.Flatten())
            fc = nn.Linear(int(input_dim), fc_dim)
            fcs.append(nn.Dropout2d(p=0.5))
            fcs.append(fc)
            fcs.append(nn.ReLU())
            input_dim = fc_dim
        return nn.Sequential(*fcs)
