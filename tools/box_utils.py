import numpy as np
import torch
from torch.autograd import Variable
EPS = 10e-20


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    Args:
        boxes: [bs, N, 4] where each row is y1, x1, y2, x2
        deltas: [bs, N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, :, 0] * height
    center_x += deltas[:, :, 1] * width
    height *= torch.exp(deltas[:, :, 2])
    width *= torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=2)
    return result


def clip_boxes(boxes, window):
    """ used also in the detection (inference) layer
    Args:
        boxes: [bs, N, 4] each col is y1, x1, y2, x2
        window: [4] in the form y1, x1, y2, x2
    """
    if window.dim() == 1:
        # for training
        boxes_out = torch.stack([
            boxes[:, :, 0].clamp(window[0].data[0], window[2].data[0]),
            boxes[:, :, 1].clamp(window[1].data[0], window[3].data[0]),
            boxes[:, :, 2].clamp(window[0].data[0], window[2].data[0]),
            boxes[:, :, 3].clamp(window[1].data[0], window[3].data[0])
        ], 2)
    elif window.dim() == 2:
        # for inference, batch size sensitive
        bs = window.size(0)
        boxes = boxes.view(bs, -1, 4)
        boxes_out = Variable(torch.zeros(boxes.size()).cuda())
        for i in range(bs):
            boxes_out[i] = torch.stack([
                boxes[i, :, 0].clamp(window[i, 0].data[0], window[i, 2].data[0]),
                boxes[i, :, 1].clamp(window[i, 1].data[0], window[i, 3].data[0]),
                boxes[i, :, 2].clamp(window[i, 0].data[0], window[i, 2].data[0]),
                boxes[i, :, 3].clamp(window[i, 1].data[0], window[i, 3].data[0])
            ], 1)
        boxes_out = boxes_out.view(-1, 4)

    return boxes_out


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


def compute_iou(boxes1, boxes2):
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size(0)), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / (union + EPS)
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps


# def np_compute_iou(box, boxes, box_area, boxes_area):
#     """Calculates IoU of the given box with the array of the given boxes.
#     box: 1D vector [y1, x1, y2, x2]
#     boxes: [boxes_count, (y1, x1, y2, x2)]
#     box_area: float. the area of 'box'
#     boxes_area: array of length boxes_count.
#     Note: the areas are passed in rather than calculated here for
#           efficency. Calculate once in the caller to avoid duplicate work.
#     """
#     # Calculate intersection areas
#     y1 = np.maximum(box[0], boxes[:, 0])
#     y2 = np.minimum(box[2], boxes[:, 2])
#     x1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[3], boxes[:, 3])
#     intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
#     union = box_area + boxes_area[:] - intersection[:]
#     iou = intersection / union
#     return iou


def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    Args:
        boxes1: [(bs, optional), N, (y1, x1, y2, x2)]
        boxes2: [(bs, optional), N, (y1, x1, y2, x2)]
    """

    # if isinstance(boxes1, np.ndarray):
    #     # numpy
    #     # Areas of anchors and GT boxes
    #     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    #     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    #     # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    #     # Each cell contains the IoU value.
    #     overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    #     for i in range(overlaps.shape[1]):
    #         box2 = boxes2[i]
    #         overlaps[:, i] = np_compute_iou(box2, boxes1, area2[i], area1)
    #     return overlaps
    #
    # else:

    # Variable
    assert boxes1.dim() == boxes2.dim()
    if boxes1.dim() == 3:
        # has bs dim
        overlaps = Variable(torch.zeros(boxes1.size(0), boxes1.size(1), boxes2.size(1)).cuda(),
                            requires_grad=False)
        for i in range(boxes1.size(0)):
            overlaps[i] = compute_iou(boxes1[i], boxes2[i])
    else:
        overlaps = compute_iou(boxes1, boxes2)

    return overlaps
