from torch.autograd import Function
from ._ext import crop_and_resize as _backend
import torch


# From Mask R-CNN paper: "We sample four regular locations, so
# that we can evaluate either max or average pooling. In fact,
# interpolating only a single value at each bin center (without
# pooling) is nearly as effective."
#
# Here we use the simplified approach of a single value per bin,
# which is how it's done in tf.crop_and_resize()
# Result: [batch * num_boxes, pool_height, pool_width, channels]
class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)

        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)
        else:
            _backend.crop_and_resize_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)

        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        else:
            _backend.crop_and_resize_backward(
                grad_outputs, boxes, box_ind, grad_image
            )

        return grad_image, None, None


# class CropAndResize(nn.Module):
#     """
#     Crop and resize ported from tensorflow
#     See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
#     """
#
#     def __init__(self, crop_height, crop_width, extrapolation_value=0):
#         super(CropAndResize, self).__init__()
#
#         self.crop_height = crop_height
#         self.crop_width = crop_width
#         self.extrapolation_value = extrapolation_value
#
#     def forward(self, image, boxes, box_ind):
#         return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)
