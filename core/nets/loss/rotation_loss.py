"""
Rotation-only pose loss for multi-spacecraft SPE3R pretraining.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from utils.postprocess import rot_6d_to_matrix

class RotationLoss(nn.Module):
    """Geodesic rotation loss evaluated on positive anchors.

    The method keeps the same calling interface as SPEEDLoss so that
    EfficientPoseHead can select either loss through the configuration.

    Translation predictions and labels are intentionally ignored.
    """

    def __init__(self):
        super(RotationLoss, self).__init__()

    def forward(self, r_raw_pr, _t_pr, R_gt, _t_gt, anchor_states):
        batch_size, num_anchors = anchor_states.shape
        positive_indices = torch.eq(anchor_states, 1)
        if not positive_indices.any():
            return r_raw_pr.sum() * 0.0
        r6d = r_raw_pr[positive_indices, :]
        R_pr = rot_6d_to_matrix(r6d)
        R_gt = R_gt.view(batch_size, 1, 3, 3).expand(-1, num_anchors, -1, -1)
        R_gt = R_gt[positive_indices, :, :]
        R_relative = torch.bmm(R_pr, R_gt.transpose(1, 2))
        trace = R_relative.diagonal(offset=0, dim1=1, dim2=2).sum(-1)
        cosine = (trace - 1.0) / 2.0
        rotation_error = torch.acos(cosine.clamp(-1.0 + 1e-06, 1.0 - 1e-06))
        return rotation_error.mean()
