
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UPLoss(nn.Module):
    """Unknown Probability Loss
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
        bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]

        # remove unknown classes
        _fg_scores = torch.cat(
            [fg_scores[:, :self.num_classes-1], fg_scores[:, -1:]], dim=1)
        _bg_scores = torch.cat(
            [bg_scores[:, :self.num_classes-1], bg_scores[:, -1:]], dim=1)

        num_fg = fg_scores.size(0)
        topk = num_fg if (self.topk == -1) or (num_fg <
                                               self.topk) else self.topk
        # use maximum entropy as a metric for uncertainty
        # we select topk proposals with maximum entropy
        if self.sampling_metric == "max_entropy":
            pos_metric = dists.Categorical(
                _fg_scores.softmax(dim=1)).entropy()
            neg_metric = dists.Categorical(
                _bg_scores.softmax(dim=1)).entropy()
        # use minimum score as a metric for uncertainty
        # we select topk proposals with minimum max-score
        elif self.sampling_metric == "min_score":
            pos_metric = -_fg_scores.max(dim=1)[0]
            neg_metric = -_bg_scores.max(dim=1)[0]
        # we randomly select topk proposals
        elif self.sampling_metric == "random":
            pos_metric = torch.rand(_fg_scores.size(0),).to(scores.device)
            neg_metric = torch.rand(_bg_scores.size(0),).to(scores.device)

        _, pos_inds = pos_metric.topk(topk)
        _, neg_inds = neg_metric.topk(topk)
        fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
        bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]

        return fg_scores, bg_scores, fg_labels, bg_labels

    def forward(self, scores: Tensor, labels: Tensor):
        fg_scores, bg_scores, fg_labels, bg_labels = self._sampling(
            scores, labels)
        # sample both fg and bg
        scores = torch.cat([fg_scores, bg_scores])
        labels = torch.cat([fg_labels, bg_labels])

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
        num_fg = fg_scores.size(0)
        targets[:num_fg, self.num_classes-2] = gt_scores[:num_fg] * \
            (1-gt_scores[:num_fg]).pow(self.alpha)
        targets[num_fg:, self.num_classes-1] = gt_scores[num_fg:] * \
            (1-gt_scores[num_fg:]).pow(self.alpha)

        return self._soft_cross_entropy(mask_scores, targets.detach())
