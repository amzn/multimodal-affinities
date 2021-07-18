# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    Credit due to: https://github.com/adambielski/
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class ContrastiveCrossEntropyLoss(nn.Module):
    """
    Contrastive Cross Entropy Loss.
    Based on: https://arxiv.org/pdf/1703.10277.pdf
    """

    def __init__(self, push_pull_weight_ratio=3, push_pull_weight_decay=0.93):
        super(ContrastiveCrossEntropyLoss, self).__init__()
        self.weight_measurer = _LossWeightMeasurer(push_pull_weight_ratio, push_pull_weight_decay)

    def forward(self, output1, output2, target, confidence):
        target = target.float()
        confidence = confidence.float()
        # weight = self.weight_measurer(target, confidence)

        euclidean_distance = F.pairwise_distance(output1, output2)
        exp_1 = (1 + torch.exp(torch.pow(euclidean_distance, 2)))
        sigmoid = 2 / exp_1
        # loss = F.binary_cross_entropy(input=sigmoid, target=target, weight=weight)
        loss = F.binary_cross_entropy(input=sigmoid, target=target)

        return loss

    def get_current_push_pull_ratio(self):
        return self.weight_measurer.push_pull_weight_ratio


class ApproximatedContrastiveCrossEntropyLoss(nn.Module):
    """
    Approximated Contrastive Cross Entropy Loss.
    Inspired by: https://arxiv.org/pdf/1703.10277.pdf
    """

    def __init__(self, push_pull_weight_ratio=3, push_pull_weight_decay=0.93):
        """
        Constructs a numerically stable approximation of ContrastiveCrossEntropyLoss
        :param push_pull_weight_ratio: A weight factor to amplify "negative" pairs (push loss).
        This scalar is multiplied by the weight of all negative pairs.
        """
        super(ApproximatedContrastiveCrossEntropyLoss, self).__init__()
        self.weight_measurer = _LossWeightMeasurer(push_pull_weight_ratio, push_pull_weight_decay)

    def forward(self, output1, output2, target, confidence):
        target = target.float()
        confidence = confidence.float()

        euclidean_distance = F.pairwise_distance(output1, output2)
        sigmoid = torch.exp(-torch.pow(euclidean_distance, 2) / 1.0)  # Approximation to sigmoid

        high_conf_ind_raw = torch.nonzero(confidence >= 1)
        high_conf_ind = high_conf_ind_raw.squeeze()
        num_high_conf_ind = high_conf_ind_raw.size()[0]

        low_conf_ind_raw = torch.nonzero(confidence < 1)
        low_conf_ind = low_conf_ind_raw.squeeze()
        num_low_conf_ind = low_conf_ind_raw.size()[0]
        loss = 0

        if num_high_conf_ind > 0:
            high_conf_sigmoid = sigmoid.index_select(0, high_conf_ind)
            high_conf_target = target.index_select(0, high_conf_ind)
            high_conf_loss = F.binary_cross_entropy(input=high_conf_sigmoid, target=high_conf_target)
            loss += 0.1 * high_conf_loss

        if num_low_conf_ind > 0:
            low_conf_sigmoid = sigmoid.index_select(0, low_conf_ind)
            low_conf_target = target.index_select(0, low_conf_ind)
            # low_conf_confidence = confidence.index_select(0, low_conf_ind)
            # weight = self.weight_measurer(low_conf_target, low_conf_confidence)
            # low_conf_loss = F.binary_cross_entropy(input=low_conf_sigmoid, target=low_conf_target, weight=weight)
            low_conf_loss = F.binary_cross_entropy(input=low_conf_sigmoid, target=low_conf_target)
            loss += low_conf_loss

        return loss

    def get_current_push_pull_ratio(self):
        return self.weight_measurer.push_pull_weight_ratio


class _LossWeightMeasurer:
    """
    Calculates all weight and confidence coeeficients together to produce a tensor of weights.
    """

    def __init__(self, push_pull_weight_ratio, push_pull_weight_decay):
        super(_LossWeightMeasurer, self).__init__()
        self.push_pull_weight_ratio_anchor = push_pull_weight_ratio
        self.push_pull_weight_ratio = push_pull_weight_ratio
        self.push_pull_weight_decay = push_pull_weight_decay

    def forward(self, target, confidence):
        """
        Considers:
        1) Positive / negative ratio in batch (target)
        2) Amplifies / reduces both push / pull loss weights according to current push_pull_weight_ratio.
        Note that both push and pull may be affected by this coefficient.
        However - pull loss weight will never drop below 1.0, while push loss weight may.
        The incentive is the further the training goes, we want embeddings to converge to single points.
        3) The confidence tensor is taken into account, per sample in batch.
        :param target: Batch of labels
        :param confidence: Tensor of confidence in labels
        :return: Tensor of weight, size of batch. Weight may surpass 1.0.
        """
        pos_amplifier = max(self.push_pull_weight_ratio_anchor - self.push_pull_weight_ratio, 1.0)
        neg_amplifier = self.push_pull_weight_ratio
        pos_weight = (float(len(torch.nonzero(target))) / len(target)) * pos_amplifier
        neg_weight = (1 - pos_weight)
        weight = pos_weight * pos_amplifier * target + neg_weight * neg_amplifier * (1 - target.float())
        weight *= confidence
        return weight

    def decay(self):
        """ Reduce ratio of push-pull loss by decay rate """
        self.push_pull_weight_ratio *= self.push_pull_weight_decay

    def __call__(self, target, confidence):
        return self.forward(target, confidence)
