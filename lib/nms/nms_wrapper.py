# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.nms.pth_nms import pth_nms
import numpy as np


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.
    used in both inference (keep_out is 1D) and 'proposal_layer' (keep_out is 2D)
    Args:
        dets:       [bs, N, 4]
        thresh:     nms threshold
    Returns:
        keep_out:   ndarray (bs, min_keep_num)
    """
    bs = dets.size(0)
    keep = []
    min_keep_num = 1e10
    for i in range(bs):
        curr_sample_keep = pth_nms(dets[i, :], thresh)
        keep.append(curr_sample_keep)
        if len(curr_sample_keep) < min_keep_num:
            min_keep_num = len(curr_sample_keep)
    keep_out = np.zeros((bs, min_keep_num), dtype=np.int32)
    for i in range(bs):
        keep_out[i, :] = keep[i][:min_keep_num].cpu().numpy()
    return keep_out
