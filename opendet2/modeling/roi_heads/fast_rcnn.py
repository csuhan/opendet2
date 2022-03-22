# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributions as dists
from detectron2.config import configurable
from detectron2.layers import (ShapeSpec, batched_nms, cat, cross_entropy,
                               nonzero_tuple)
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                     _log_classification_stats)
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.structures.boxes import matched_boxlist_iou
#  fast_rcnn_inference)
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from ..layers import MLP
from ..losses import ICLoss, UPLoss

ROI_BOX_OUTPUT_LAYERS_REGISTRY = Registry("ROI_BOX_OUTPUT_LAYERS")
ROI_BOX_OUTPUT_LAYERS_REGISTRY.__doc__ = """
ROI_BOX_OUTPUT_LAYERS
"""


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float = 1.0,
):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
):
    valid_mask = torch.isfinite(boxes).all(
        dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    # apply nms between known classes and unknown class for visualization.
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(
            boxes, scores, filter_inds, iou_thr=vis_iou_thr)

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def unknown_aware_nms(boxes, scores, labels, ukn_class_id=80, iou_thr=0.9):
    u_inds = labels[:, 1] == ukn_class_id
    k_inds = ~u_inds
    if k_inds.sum() == 0 or u_inds.sum() == 0:
        return boxes, scores, labels

    k_boxes, k_scores, k_labels = boxes[k_inds], scores[k_inds], labels[k_inds]
    u_boxes, u_scores, u_labels = boxes[u_inds], scores[u_inds], labels[u_inds]

    ious = pairwise_iou(Boxes(k_boxes), Boxes(u_boxes))
    mask = torch.ones((ious.size(0), ious.size(1), 2), device=ious.device)
    inds = (ious > iou_thr).nonzero()
    if not inds.numel():
        return boxes, scores, labels

    for [ind_x, ind_y] in inds:
        if k_scores[ind_x] >= u_scores[ind_y]:
            mask[ind_x, ind_y, 1] = 0
        else:
            mask[ind_x, ind_y, 0] = 0

    k_inds = mask[..., 0].mean(dim=1) == 1
    u_inds = mask[..., 1].mean(dim=0) == 1

    k_boxes, k_scores, k_labels = k_boxes[k_inds], k_scores[k_inds], k_labels[k_inds]
    u_boxes, u_scores, u_labels = u_boxes[u_inds], u_scores[u_inds], u_labels[u_inds]

    boxes = torch.cat([k_boxes, u_boxes])
    scores = torch.cat([k_scores, u_scores])
    labels = torch.cat([k_labels, u_labels])

    return boxes, scores, labels


logger = logging.getLogger(__name__)


def build_roi_box_output_layers(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS
    return ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(name)(cfg, input_shape)


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class CosineFastRCNNOutputLayers(FastRCNNOutputLayers):

    @configurable
    def __init__(
        self,
        *args,
        scale: int = 20,
        vis_iou_thr: float = 1.0,
        **kargs,

    ):
        super().__init__(*args, **kargs)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(
            self.cls_score.in_features, self.num_classes + 1, bias=False)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        # scaling factor
        self.scale = scale
        self.vis_iou_thr = vis_iou_thr

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['scale'] = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        ret['vis_iou_thr'] = cfg.MODEL.ROI_HEADS.VIS_IOU_THRESH
        return ret

    def forward(self, feats):

        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        return scores, proposal_deltas

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):

        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.vis_iou_thr,
        )

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        if not len(proposals):
            return []
        proposal_deltas = predictions[1]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat(
            [p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class OpenDetFastRCNNOutputLayers(CosineFastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        *args,
        num_known_classes,
        max_iters,
        up_loss_start_iter,
        up_loss_sampling_metric,
        up_loss_topk,
        up_loss_alpha,
        up_loss_weight,
        ic_loss_out_dim,
        ic_loss_queue_size,
        ic_loss_in_queue_size,
        ic_loss_batch_iou_thr,
        ic_loss_queue_iou_thr,
        ic_loss_queue_tau,
        ic_loss_weight,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.num_known_classes = num_known_classes
        self.max_iters = max_iters

        self.up_loss = UPLoss(
            self.num_classes,
            sampling_metric=up_loss_sampling_metric,
            topk=up_loss_topk,
            alpha=up_loss_alpha
        )
        self.up_loss_start_iter = up_loss_start_iter
        self.up_loss_weight = up_loss_weight

        self.encoder = MLP(self.cls_score.in_features, ic_loss_out_dim)
        self.ic_loss_loss = ICLoss(tau=ic_loss_queue_tau)
        self.ic_loss_out_dim = ic_loss_out_dim
        self.ic_loss_queue_size = ic_loss_queue_size
        self.ic_loss_in_queue_size = ic_loss_in_queue_size
        self.ic_loss_batch_iou_thr = ic_loss_batch_iou_thr
        self.ic_loss_queue_iou_thr = ic_loss_queue_iou_thr
        self.ic_loss_weight = ic_loss_weight

        self.register_buffer('queue', torch.zeros(
            self.num_known_classes, ic_loss_queue_size, ic_loss_out_dim))
        self.register_buffer('queue_label', torch.empty(
            self.num_known_classes, ic_loss_queue_size).fill_(-1).long())
        self.register_buffer('queue_ptr', torch.zeros(
            self.num_known_classes, dtype=torch.long))

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'num_known_classes': cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "max_iters": cfg.SOLVER.MAX_ITER,

            "up_loss_start_iter": cfg.UPLOSS.START_ITER,
            "up_loss_sampling_metric": cfg.UPLOSS.SAMPLING_METRIC,
            "up_loss_topk": cfg.UPLOSS.TOPK,
            "up_loss_alpha": cfg.UPLOSS.ALPHA,
            "up_loss_weight": cfg.UPLOSS.WEIGHT,

            "ic_loss_out_dim": cfg.ICLOSS.OUT_DIM,
            "ic_loss_queue_size": cfg.ICLOSS.QUEUE_SIZE,
            "ic_loss_in_queue_size": cfg.ICLOSS.IN_QUEUE_SIZE,
            "ic_loss_batch_iou_thr": cfg.ICLOSS.BATCH_IOU_THRESH,
            "ic_loss_queue_iou_thr": cfg.ICLOSS.QUEUE_IOU_THRESH,
            "ic_loss_queue_tau": cfg.ICLOSS.TEMPERATURE,
            "ic_loss_weight": cfg.ICLOSS.WEIGHT,

        })
        return ret

    def forward(self, feats):
        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        # encode feature with MLP
        mlp_feat = self.encoder(cls_x)

        return scores, proposal_deltas, mlp_feat

    def get_up_loss(self, scores, gt_classes):
        # start up loss after several warmup iters
        storage = get_event_storage()
        if storage.iter > self.up_loss_start_iter:
            loss_cls_up = self.up_loss(scores, gt_classes)
        else:
            loss_cls_up = scores.new_tensor(0.0)

        return {"loss_cls_up": self.up_loss_weight * loss_cls_up}

    def get_ic_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > self.ic_loss_batch_iou_thr) & (
            gt_classes != self.num_classes)
        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)
        # loss decay
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        return {"loss_cls_ic": self.ic_loss_weight * decay_weight * loss_ic_loss}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        # 1. gather variable
        feat = self.concat_all_gather(feat)
        gt_classes = self.concat_all_gather(gt_classes)
        ious = self.concat_all_gather(ious)
        # 2. filter by iou and obj, remove bg
        keep = (ious > iou_thr) & (gt_classes != self.num_classes)
        feat, gt_classes = feat[keep], gt_classes[keep]

        for i in range(self.num_known_classes):
            ptr = int(self.queue_ptr[i])
            cls_ind = gt_classes == i
            cls_feat, cls_gt_classes = feat[cls_ind], gt_classes[cls_ind]
            # 3. sort by similarity, low sim ranks first
            cls_queue = self.queue[i, self.queue_label[i] != -1]
            _, sim_inds = F.cosine_similarity(
                cls_feat[:, None], cls_queue[None, :], dim=-1).mean(dim=1).sort()
            top_sim_inds = sim_inds[:self.ic_loss_in_queue_size]
            cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]
            # 4. in queue
            batch_size = cls_feat.size(
                0) if ptr + cls_feat.size(0) <= self.ic_loss_queue_size else self.ic_loss_queue_size - ptr
            self.queue[i, ptr:ptr+batch_size] = cls_feat[:batch_size]
            self.queue_label[i, ptr:ptr + batch_size] = cls_gt_classes[:batch_size]

            ptr = ptr + batch_size if ptr + batch_size < self.ic_loss_queue_size else 0
            self.queue_ptr[i] = ptr

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        world_size = comm.get_world_size()
        # single GPU, directly return the tensor
        if world_size == 1:
            return tensor
        # multiple GPUs, gather tensors
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    def losses(self, predictions, proposals, input_features=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, mlp_feat = predictions
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(
                proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes")
                  else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls_ce": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

        # up loss
        losses.update(self.get_up_loss(scores, gt_classes))

        ious = cat([p.iou for p in proposals], dim=0)
        # we first store feats in the queue, then cmopute loss
        self._dequeue_and_enqueue(
            mlp_feat, gt_classes, ious, iou_thr=self.ic_loss_queue_iou_thr)
        losses.update(self.get_ic_loss(mlp_feat, gt_classes, ious))

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class PROSERFastRCNNOutputLayers(CosineFastRCNNOutputLayers):
    """PROSER
    """
    @configurable
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.proser_weight = 0.1

    def get_proser_loss(self, scores, gt_classes):
        num_sample, num_classes = scores.shape
        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != gt_classes[:, None].repeat(1, num_classes)
        mask = mask[inds].reshape(num_sample, num_classes-1)
        mask_scores = torch.gather(scores, 1, mask)

        targets = torch.zeros_like(gt_classes)
        fg_inds = gt_classes != self.num_classes
        targets[fg_inds] = self.num_classes-2
        targets[~fg_inds] = self.num_classes-1

        loss_cls_proser = cross_entropy(mask_scores, targets)
        return {"loss_cls_proser": self.proser_weight * loss_cls_proser}

    def losses(self, predictions, proposals, input_features=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(
                proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes")
                  else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls_ce": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        losses.update(self.get_proser_loss(scores, gt_classes))

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class DropoutFastRCNNOutputLayers(CosineFastRCNNOutputLayers):
    
    @configurable
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.dropout = nn.Dropout(p=0.5)
        self.entropy_thr = 0.25

    def forward(self, feats, testing=False):
        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        if testing:
            self.dropout.train()
            x_normalized = self.dropout(x_normalized)
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        return scores, proposal_deltas

    def inference(self, predictions: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions[0], proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_probs(
        self, predictions: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        # mean of multiple observations
        scores = torch.stack([pred[0] for pred in predictions], dim=-1)
        scores = scores.mean(dim=-1)
        # threshlod by entropy
        norm_entropy = dists.Categorical(scores.softmax(
            dim=1)).entropy() / np.log(self.num_classes)
        inds = norm_entropy > self.entropy_thr
        max_scores = scores.max(dim=1)[0]
        # set those with high entropy unknown objects
        scores[inds, :] = 0.0
        scores[inds, self.num_classes-1] = max_scores[inds]

        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
