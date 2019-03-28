import random
import numpy as np
import scipy.misc
import scipy.ndimage
from tools.box_utils import extract_bboxes


def compose_image_meta(image_id, image_shape, window, active_class_ids, coco_image_id):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                # size=1
        list(image_shape) +         # size=3
        list(window) +              # size=4 (y1, x1, y2, x2) in image coordinates
        list(active_class_ids) +    # size=num_classes
        [coco_image_id]             # size=1
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:-1]
    coco_image_id = meta[:, -1]
    return image_id, image_shape, window, active_class_ids, coco_image_id


# def parse_image_meta_graph(meta):
#     """Parses a tensor that contains image attributes to its components.
#     See compose_image_meta() for more details.
#
#     meta: [batch, meta length] where meta length depends on NUM_CLASSES
#     """
#     image_id = meta[:, 0]
#     image_shape = meta[:, 1:4]
#     window = meta[:, 4:8]
#     active_class_ids = meta[:, 8:]
#     return [image_id, image_shape, window, active_class_ids]


# def mold_image(images, config):
#     """Takes RGB images with 0-255 values and subtraces
#     the mean pixel and converts it to float. Expects image
#     colors in RGB order.
#     """
#     return images.astype(np.float32) - config.MEAN_PIXEL
#
#
# def unmold_image(normalized_images, config):
#     """Takes a image normalized with mold() and returns the original."""
#     return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    Args:
        image
        min_dim:    if provided, resizes the image such that its smaller dimension == min_dim
        max_dim:    if provided, ensures that the image longest side doesn't exceed this value.

        padding:    if true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
        image:      the resized image
        window:     (y1, x1, y2, x2). If max_dim is provided, padding might
                            be inserted in the returned image. If so, this window is the
                            coordinates of the image part of the full image (excluding
                            the padding). The x2, y2 pixels are not included.
        scale:      The scale factor used to resize the image
        padding:    Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    # SUPER IMPORTANT: all images are forced to resize to MAX_DIM via padding
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # TODO: erase user warning
    mask = scipy.ndimage.zoom(mask, (scale, scale, 1), order=3)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()
    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size != 0:
            # raise Exception("Invalid bounding box with area of zero")
            m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
            mini_mask[:, :, i] = np.where(m >= 128, 1, 0)

    return mini_mask


# def expand_mask(bbox, mini_mask, image_shape):
#     """Resizes mini masks back to image size. Reverses the change
#     of minimize_mask().
#
#     See inspect_data.ipynb notebook for more details.
#     """
#     mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
#     for i in range(mask.shape[-1]):
#         m = mini_mask[:, :, i]
#         y1, x1, y2, x2 = bbox[i][:4]
#         h = y2 - y1
#         w = x2 - x1
#         m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
#         mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
#     return mask


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Data Generator (called in __get_item__)
############################################################
def load_image_and_gt(dataset, config, image_id, augment=False, use_mini_mask=False):
    """Load and return ground truth datasets for an image (image, mask, bounding boxes).

    augment:        If true, apply random image augmentation.
    use_mini_mask:  If False, returns full-size masks that are the same height
                        and width as the original image. These can be big, for example
                        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
                        224x224 and are generated by extracting the bounding box of the
                        object and resizing it to MINI_MASK_SHAPE.
    Returns:
    image:          [height, width, 3]
    shape:          the original shape of the image before resizing and cropping.
    class_ids:      [instance_count] Integer class IDs
    bbox:           [instance_count, (y1, x1, y2, x2)]
    mask:           [height, width, instance_count]. The height and width are those
                    of the image unless use_mini_mask is True, in which case they are
                    defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    image, window, scale, padding = \
        resize_image(image, min_dim=config.DATA.IMAGE_MIN_DIM,
                     max_dim=config.DATA.IMAGE_MAX_DIM, padding=config.DATA.IMAGE_PADDING)
    mask = resize_mask(mask, scale, padding)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bbox, mask, config.MRCNN.MINI_MASK_SHAPE)

    # Image meta datasets
    coco_image_id = dataset.image_info[image_id]["id"]
    image_meta = compose_image_meta(image_id, image.shape, window, active_class_ids, coco_image_id)

    return image, image_meta, class_ids, bbox, mask