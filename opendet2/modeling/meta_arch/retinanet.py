# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributions as dists
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.meta_arch.retinanet import RetinaNet, RetinaNetHead
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit
from torch import Tensor, nn
from torch.nn import functional as F

from ..layers import ConvMLP
from ..losses import ICLoss

logger = logging.getLogger(__name__)


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class UPLoss(nn.Module):
    """Unknown Probability Loss for RetinaNet
    """

    def __init__(self,
                 num_classes: int,
                 sampling_metric: str = "min_score",
                 topk: int = 3,
                 alpha: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        assert sampling_metric in ["min_score", "max_entropy", "random"]
        self.sampling_metric = sampling_metric
        # if topk==-1, sample len(fg)*2 examples
        self.topk = topk
        self.alpha = alpha

    def _soft_cross_entropy(self, input: Tensor, target: Tensor):
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def _sampling(self, scores: Tensor, labels: Tensor):
        fg_inds = labels != self.num_classes
        fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]

        # remove unknown classes
        _fg_scores = torch.cat(
            [fg_scores[:, :self.num_classes-1], fg_scores[:, -1:]], dim=1)

        num_fg = fg_scores.size(0)
        topk = num_fg if (self.topk == -1) or (num_fg <
                                               self.topk) else self.topk
        # use maximum entropy as a metric for uncertainty
        # we select topk proposals with maximum entropy
        if self.sampling_metric == "max_entropy":
            pos_metric = dists.Categorical(
                _fg_scores.softmax(dim=1)).entropy()
        # use minimum score as a metric for uncertainty
        # we select topk proposals with minimum max-score
        elif self.sampling_metric == "min_score":
            pos_metric = -_fg_scores.max(dim=1)[0]
        # we randomly select topk proposals
        elif self.sampling_metric == "random":
            pos_metric = torch.rand(_fg_scores.size(0),).to(scores.device)

        _, pos_inds = pos_metric.topk(topk)
        fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]

        return fg_scores, fg_labels

    def forward(self, scores: Tensor, labels: Tensor):
        scores, labels = self._sampling(scores, labels)

        num_sample, num_classes = scores.shape
        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != labels[:, None].repeat(1, num_classes)
        mask = mask[inds].reshape(num_sample, num_classes-1)

        gt_scores = torch.gather(
            F.softmax(scores, dim=1), 1, labels[:, None]).squeeze(1)
        mask_scores = torch.gather(scores, 1, mask)

        gt_scores[gt_scores < 0] = 0.0
        targets = torch.zeros_like(mask_scores)
        targets[:, self.num_classes-2] = gt_scores * \
            (1-gt_scores).pow(self.alpha)

        return self._soft_cross_entropy(mask_scores, targets.detach())


@META_ARCH_REGISTRY.register()
class OpenSetRetinaNet(RetinaNet):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    @configurable
    def __init__(
        self,
        num_known_classes,
        max_iters,
        up_loss_start_iter,
        up_loss_sampling_metric,
        up_loss_topk,
        up_loss_alpha,
        up_loss_weight,
        ins_con_out_dim,
        ins_con_queue_size,
        ins_con_in_queue_size,
        ins_con_batch_iou_thr,
        ins_con_queue_iou_thr,
        ins_con_queue_tau,
        ins_con_loss_weight,
        *args,
        **kargs,
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

        self.ins_con_loss = ICLoss(tau=ins_con_queue_tau)
        self.ins_con_out_dim = ins_con_out_dim
        self.ins_con_queue_size = ins_con_queue_size
        self.ins_con_in_queue_size = ins_con_in_queue_size
        self.ins_con_batch_iou_thr = ins_con_batch_iou_thr
        self.ins_con_queue_iou_thr = ins_con_queue_iou_thr
        self.ins_con_loss_weight = ins_con_loss_weight

        self.register_buffer('queue', torch.zeros(
            self.num_known_classes, ins_con_queue_size, ins_con_out_dim))
        self.register_buffer('queue_label', torch.empty(
            self.num_known_classes, ins_con_queue_size).fill_(-1).long())
        self.register_buffer('queue_ptr', torch.zeros(
            self.num_known_classes, dtype=torch.long))

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        backbone_shape = ret["backbone"].output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        head = OpenSetRetinaNetHead(cfg, feature_shapes)
        ret.update({
            "head": head,
            "num_known_classes": cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "max_iters": cfg.SOLVER.MAX_ITER,

            "up_loss_start_iter": cfg.UPLOSS.START_ITER,
            "up_loss_sampling_metric": cfg.UPLOSS.SAMPLING_METRIC,
            "up_loss_topk": cfg.UPLOSS.TOPK,
            "up_loss_alpha": cfg.UPLOSS.ALPHA,
            "up_loss_weight": cfg.UPLOSS.WEIGHT,

            "ins_con_out_dim": cfg.ICLOSS.OUT_DIM,
            "ins_con_queue_size": cfg.ICLOSS.QUEUE_SIZE,
            "ins_con_in_queue_size": cfg.ICLOSS.IN_QUEUE_SIZE,
            "ins_con_batch_iou_thr": cfg.ICLOSS.BATCH_IOU_THRESH,
            "ins_con_queue_iou_thr": cfg.ICLOSS.QUEUE_IOU_THRESH,
            "ins_con_queue_tau": cfg.ICLOSS.TEMPERATURE,
            "ins_con_loss_weight": cfg.ICLOSS.WEIGHT,
        })
        return ret

    def get_up_loss(self, scores, gt_classes):
        # start up loss after warmup iters
        storage = get_event_storage()
        if storage.iter > self.up_loss_start_iter:
            loss_cls_up = self.up_loss(scores, gt_classes)
        else:
            loss_cls_up = scores.new_tensor(0.0)

        return self.up_loss_weight * loss_cls_up

    def get_ins_con_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > self.ins_con_batch_iou_thr) & (
            gt_classes != self.num_classes)

        if not pos_inds.sum():
            return feat.new_tensor(0.0)

        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ins_con_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_ins_con = self.ins_con_loss(feat, gt_classes, queue, queue_label)
        # loss decay
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        return self.ins_con_loss_weight * decay_weight * loss_ins_con

    @ torch.no_grad()
    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        # 1. gather variable
        # feat = self.concat_all_gather(feat)
        # gt_classes = self.concat_all_gather(gt_classes)
        # ious = self.concat_all_gather(ious)
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
            top_sim_inds = sim_inds[:self.ins_con_in_queue_size]
            cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]
            # 4. in queue
            batch_size = cls_feat.size(
                0) if ptr + cls_feat.size(0) <= self.ins_con_queue_size else self.ins_con_queue_size - ptr
            self.queue[i, ptr:ptr+batch_size] = cls_feat[:batch_size]
            self.queue_label[i, ptr:ptr +
                             batch_size] = cls_gt_classes[:batch_size]

            ptr = ptr + batch_size if ptr + batch_size < self.ins_con_queue_size else 0
            self.queue_ptr[i] = ptr

    @ torch.no_grad()
    def concat_all_gather(self, tensor):
        tensors_gather = [torch.ones_like(tensor) for _ in range(
            torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas, pred_mlp_feats = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(
            x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(
            x, 4) for x in pred_anchor_deltas]
        pred_mlp_feats = [permute_to_N_HWA_K(
            x, self.ins_con_out_dim) for x in pred_mlp_feats]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]

            gt_labels, gt_boxes, gt_ious = self.label_anchors(
                anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, pred_mlp_feats,
                                 gt_labels, pred_anchor_deltas, gt_boxes, gt_ious)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(
                anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, anchors, pred_logits, pred_mlp_feats, gt_labels, pred_anchor_deltas, gt_boxes, gt_ious):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class

        loss_cls_ce = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_cls_up = self.get_up_loss(cat(pred_logits, dim=1)[
                                       valid_mask], gt_labels[valid_mask])

        gt_ious = torch.stack(gt_ious)
        # we first store feats in the queue, then cmopute the loss
        pred_mlp_feats = cat(pred_mlp_feats, dim=1)[valid_mask]  # [N, *, 128]
        # [N*, 128]
        pred_mlp_feats = pred_mlp_feats.reshape(-1, pred_mlp_feats.shape[-1])
        self._dequeue_and_enqueue(
            pred_mlp_feats, gt_labels[valid_mask], gt_ious[valid_mask], iou_thr=self.ins_con_queue_iou_thr)
        loss_ins_con = self.get_ins_con_loss(
            pred_mlp_feats, gt_labels[valid_mask], gt_ious[valid_mask])

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        return {
            "loss_cls_ce": loss_cls_ce / self.loss_normalizer,
            "loss_box_reg": loss_box_reg / self.loss_normalizer,
            "loss_ins_con": loss_ins_con,
            "loss_cls_up": loss_cls_up,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):

        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        matched_gt_ious = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(
                match_quality_matrix)
            # del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                matched_gt_ious_i = match_quality_matrix.max(dim=1)[
                    0][matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                matched_gt_ious_i = torch.zeros_like(matched_idxs)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_ious.append(matched_gt_ious_i)

            del match_quality_matrix

        return gt_labels, matched_gt_boxes, matched_gt_ious


class OpenSetRetinaNetHead(RetinaNetHead):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self,
        *args,
        ins_con_out_dim,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.mlp = ConvMLP(kargs["conv_dims"][-1], ins_con_out_dim * kargs["num_anchors"])

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["ins_con_out_dim"] = cfg.ICLOSS.OUT_DIM
        return ret

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        mlp_feats = []
        bbox_reg = []
        for feature in features:
            cls_feat = self.cls_subnet(feature)
            mlp_feats.append(self.mlp(cls_feat))
            logits.append(self.cls_score(cls_feat))

            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg, mlp_feats
