import random
import os
from tools.utils import merge_cfg_from_file, merge_cfg_from_list, print_log
import math
from tools.collections import AttrDict
import torch
import numpy as np

# Pre-defined layer regular expressions
LAYER_REGEX = {
    # only heads
    "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|"
             r"(rpn.*)|(classifier.*)|(mask.*)|(dev_roi.*)|(ot_loss.*)|(fpn.*\_ot.*)",

    # From a specific resnet stage and up
    "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|"
          r"(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(dev_roi.*)|(ot_loss.*)|(fpn.*\_ot.*)",

    "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|"
          r"(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(dev_roi.*)|(ot_loss.*)|(fpn.*\_ot.*)",

    "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|"
          r"(rpn.*)|(classifier.*)|(mask.*)|(dev_roi.*)|(ot_loss.*)|(fpn.*\_ot.*)",
    # All layers
    "all": ".*",
}

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

TEMP = {'heads': 1, '4+': 2, 'all': 3}


class Config(object):
    # ==================================
    MODEL = AttrDict()
    # Path to pretrained imagenet model
    MODEL.PRETRAIN_IMAGENET_MODEL = os.path.join('datasets/pretrain_model', "resnet50_imagenet.pth")
    # Path to pretrained weights file
    MODEL.PRETRAIN_COCO_MODEL = os.path.join('datasets/pretrain_model', 'mask_rcnn_coco.pth')
    MODEL.INIT_FILE_CHOICE = 'last'  # or file (xxx.pth)
    MODEL.INIT_MODEL = None   # set in 'utils.py'
    MODEL.BACKBONE = 'resnet101'
    MODEL.BACKBONE_STRIDES = []
    MODEL.BACKBONE_SHAPES = []

    # ==================================
    DATASET = AttrDict()
    # Number of classification classes (including background)
    DATASET.NUM_CLASSES = 81
    DATASET.YEAR = '2014'
    DATASET.PATH = 'datasets/coco'

    # ==================================
    RPN = AttrDict()
    # Length of square anchor side in pixels
    RPN.ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN.ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on (stride=2,3,4...).
    RPN.ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more proposals.
    RPN.NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN.TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum suppression for RPN part
    RPN.PRE_NMS_LIMIT = 6000
    RPN.POST_NMS_ROIS_TRAINING = 2000
    RPN.POST_NMS_ROIS_INFERENCE = 1000

    RPN.TARGET_POS_THRES = .7
    RPN.TARGET_NEG_THRES = .3

    # ==================================
    MRCNN = AttrDict()
    # If enabled, resize instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    MRCNN.USE_MINI_MASK = True
    MRCNN.MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    # Pooled ROIs
    MRCNN.POOL_SIZE = 7         # cls/bbox stream
    MRCNN.MASK_POOL_SIZE = 14   # mask stream
    MRCNN.MASK_SHAPE = [28, 28]

    # ==================================
    DATA = AttrDict()
    # Input image resize
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    DATA.IMAGE_MIN_DIM = 800
    DATA.IMAGE_MAX_DIM = 1024
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    DATA.IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    DATA.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Maximum number of ground truth instances to use in one image
    DATA.MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    DATA.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    DATA.IMAGE_SHAPE = []
    # Quote from "roytseng-tw/Detectron.pytorch":
    # Number of Python threads to use for the data loader (warning: using too many
    # threads can cause GIL-based interference with Python Ops leading to *slower*
    # training; 4 seems to be the sweet spot in our experience)
    DATA.LOADER_WORKER_NUM = 2

    # ==================================
    ROIS = AttrDict()
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting the RPN NMS threshold.
    ROIS.TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROIS.ROI_POSITIVE_RATIO = 0.33
    # Eqn.(1) in FPN paper
    # useless when DEV.ASSIGN_BOX_ON_ALL_SCALE is True
    ROIS.ASSIGN_ANCHOR_BASE = 224.
    ROIS.METHOD = 'roi_align'  # or roi_pool

    # ==================================
    TEST = AttrDict()
    TEST.BATCH_SIZE = 0   # set in _set_value()
    # Max number of final detections
    TEST.DET_MAX_INSTANCES = 100
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    TEST.DET_MIN_CONFIDENCE = 0
    # Non-maximum suppression threshold for detection
    TEST.DET_NMS_THRESHOLD = 0.3
    TEST.SAVE_IM = False

    # ==================================
    TRAIN = AttrDict()
    TRAIN.BATCH_SIZE = 6
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer implementation.
    TRAIN.OPTIM_METHOD = 'sgd'
    TRAIN.INIT_LR = 0.01
    TRAIN.MOMENTUM = 0.9
    # Weight decay regularization
    TRAIN.WEIGHT_DECAY = 0.0001
    TRAIN.GAMMA = 0.1
    TRAIN.LR_POLICY = 'steps_with_decay'
    TRAIN.END2END = False
    # in epoch
    TRAIN.SCHEDULE = [6, 4, 3]
    TRAIN.LR_WARM_UP = False
    TRAIN.LR_WP_ITER = 500
    TRAIN.LR_WP_FACTOR = 1. / 3.

    TRAIN.CLIP_GRAD = True
    TRAIN.MAX_GRAD_NORM = 5.0

    # let bn learn and also apply the same weight decay when setting up optimizer
    TRAIN.BN_LEARN = False

    # evaluate mAP after each stage
    TRAIN.DO_VALIDATION = True
    TRAIN.SAVE_FREQ_WITHIN_EPOCH = 10
    TRAIN.FORCE_START_EPOCH = 0   # when you resume training and change the batch size, this is useful
    # apply OT loss in FPN heads
    TRAIN.FPN_OT_LOSS = False
    TRAIN.FPN_OT_LOSS_FAC = 1.

    # ==============================
    DEV = AttrDict()
    DEV.SWITCH = False
    DEV.INIT_BUFFER_WEIGHT = 'scratch'    # currently only support this
    # set to 1 if use all historic data
    DEV.BUFFER_SIZE = 1000
    # set to <= 0 if trained from the very first iter
    DEV.EFFECT_AFER_EP_PERCENT = 0.

    DEV.MULTI_UPSAMPLER = False   # does not affect much
    # if 1, standard conv
    DEV.UPSAMPLE_FAC = 2.

    DEV.LOSS_CHOICE = 'l1'
    DEV.OT_ONE_DIM_FORM = 'conv'   # effective if loss_choice is 'ot'
    DEV.LOSS_FAC = 0.5
    # compute meta_los of small boxes on an instance or class level
    DEV.INST_LOSS = False

    DEV.FEAT_BRANCH_POOL_SIZE = 14
    # ignore regression loss (only for **DEBUG**);
    # doomed if you use it during deployment
    DEV.DIS_REG_LOSS = False

    # assign anchors on all scales and split anchor based on roi-pooling output size
    # if used, then ROIS.ASSIGN_ANCHOR_BASE is inactivated
    DEV.ASSIGN_BOX_ON_ALL_SCALE = False
    # provide a baseline (no meta_loss) to compare
    DEV.BASELINE = False

    DEV.BIG_SUPERVISE = False
    DEV.BIG_LOSS_CHOICE = 'ce'      # default setting (currently only support this)
    DEV.BIG_FC_INIT = 'scratch'     # or 'coco_pretrain'
    DEV.BIG_LOSS_FAC = 1.
    DEV.BIG_FC_INIT_LIST = dict()

    DEV.STRUCTURE = 'alpha'   # 'beta'
    DEV.DIS_UPSAMPLER = False
    DEV.BIG_FEAT_DETACH = True
    # merge compare_feat output into classifier
    DEV.CLS_MERGE_FEAT = False
    DEV.CLS_MERGE_MANNER = 'simple_add'   # 'linear_add'
    DEV.CLS_MERGE_FAC = .5

    # ==============================
    CTRL = AttrDict()
    CTRL.CONFIG_NAME = ''
    CTRL.PHASE = ''
    CTRL.DEBUG = None
    # train on minival and test also on minival
    CTRL.QUICK_VERIFY = False

    CTRL.SHOW_INTERVAL = 50
    CTRL.PROFILE_ANALYSIS = False  # show time for some pass

    # ==============================
    TSNE = AttrDict()
    TSNE.SKIP_INFERENCE = True    # skip the evaluation (compute mAP)
    TSNE.A_FEW = False
    TSNE.PERPLEXITY = 30
    TSNE.METRIC = 'euclidean'
    TSNE.N_TOPICS = 2
    TSNE.BATCH_SZ = 1024     # 1024    # bigger bs is more sparse
    TSNE.TOTAL_EP = 150
    TSNE.ELLIPSE = True
    TSNE.SAMPLE_CHOICE = 'set1'   # for detailed config, see 'def prepare_data()' in tools/tsne/prepare_data.py
    TSNE.FIG_FOLDER_SUX = 'debug5'   # custom folder name

    # ==============================
    MISC = AttrDict()
    MISC.SEED = 2000
    MISC.USE_VISDOM = False
    MISC.VIS = AttrDict()
    MISC.VIS.PORT = -1  # must be passed from configs on different servers
    # the following will be set somewhere else
    MISC.LOG_FILE = None
    MISC.DET_RESULT_FILE = None
    MISC.SAVE_IMAGE_DIR = None
    MISC.RESULT_FOLDER = None
    MISC.DEVICE_ID = []
    MISC.GPU_COUNT = -1

    def display(self, log_file, quiet=False):
        """Display *final* configuration values."""
        print_log("Configurations:", file=log_file, quiet_termi=quiet)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                value = getattr(self, a)
                if isinstance(value, AttrDict):
                    print_log("{}:".format(a), log_file, quiet_termi=quiet)
                    for _, key in enumerate(value):
                        print_log("\t{:30}\t\t{}".format(key, value[key]), log_file, quiet_termi=quiet)
                else:
                    print_log("{}\t{}".format(a, value), log_file, quiet_termi=quiet)
        print_log("\n", log_file, quiet_termi=quiet)

    def _set_value(self):
        """Set values of computed attributes. Override all previous settings."""

        random.seed(self.MISC.SEED)
        torch.manual_seed(self.MISC.SEED)

        if self.CTRL.QUICK_VERIFY:
            self.CTRL.SHOW_INTERVAL = 5
            self.TRAIN.SAVE_FREQ_WITHIN_EPOCH = 2

        if self.CTRL.DEBUG:
            self.CTRL.SHOW_INTERVAL = 1
            self.DATA.IMAGE_MIN_DIM = 320
            self.DATA.IMAGE_MAX_DIM = 512
            self.CTRL.PROFILE_ANALYSIS = False
            self.TSNE.A_FEW = True

        # set MISC.RESULT_FOLDER, 'results/base_101/train (or inference)/'
        self.MISC.RESULT_FOLDER = os.path.join(
            'results', self.CTRL.CONFIG_NAME.lower(), self.CTRL.PHASE)
        if not os.path.exists(self.MISC.RESULT_FOLDER):
            os.makedirs(self.MISC.RESULT_FOLDER)

        self.TEST.BATCH_SIZE = 2 * self.TRAIN.BATCH_SIZE

        # MUST be left **at the end**
        # The strides of each layer of the FPN Pyramid.
        if self.MODEL.BACKBONE == 'resnet101':
            self.MODEL.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
        else:
            raise Exception('unknown backbone structure')

        # Input image size
        self.DATA.IMAGE_SHAPE = np.array(
            [self.DATA.IMAGE_MAX_DIM, self.DATA.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size
        self.MODEL.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.DATA.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.DATA.IMAGE_SHAPE[1] / stride))]
             for stride in self.MODEL.BACKBONE_STRIDES])

        if self.MISC.USE_VISDOM:
            if self.CTRL.DEBUG:
                self.MISC.VIS.PORT = 2042

            assert self.MISC.VIS.PORT > 0, 'vis_port not designated!!!'

            print('\n[visdom is activated] remember to execute '
                  '**python -m visdom.server -port={:d}** on server (or pc)!\n'.format(self.MISC.VIS.PORT))
            self.MISC.VIS.LINE = 100
            self.MISC.VIS.TXT = 200
            self.MISC.VIS.IMG = 300
            self.MISC.VIS.LOSS_LEGEND = [
                'total_loss', 'rpn_cls', 'rpn_bbox',
                'mrcnn_cls', 'mrcnn_bbox', 'mrcnn_mask_loss']
            if self.DEV.SWITCH and not self.DEV.BASELINE:
                self.MISC.VIS.LOSS_LEGEND.append('meta_loss')
            if self.DEV.SWITCH and self.DEV.BIG_SUPERVISE:
                self.MISC.VIS.LOSS_LEGEND.append('big_loss')
            if self.TRAIN.FPN_OT_LOSS:
                self.MISC.VIS.LOSS_LEGEND.append('fpn_ot_loss')

        if self.MISC.GPU_COUNT == 8:
            self.DATA.LOADER_WORKER_NUM = 32
        elif self.MISC.GPU_COUNT == 4:
            self.DATA.LOADER_WORKER_NUM = 16

        if self.DEV.BIG_FC_INIT == 'coco_pretrain':
            self.DEV.BIG_FC_INIT_LIST = {
                # target network vs pretrain network
                'dev_roi.big_fc_layer.weight': 'classifier.linear_class.weight',
                'dev_roi.big_fc_layer.bias': 'classifier.linear_class.bias',
            }
        # TODO (low): add more here; delete some config for brevity
        if not self.TRAIN.LR_WARM_UP:
            del self.TRAIN['LR_WP_ITER']
            del self.TRAIN['LR_WP_FACTOR']
        if not self.DEV.BIG_SUPERVISE:
            del self.DEV['BIG_LOSS_FAC']
            del self.DEV['BIG_FC_INIT']
            del self.DEV['BIG_LOSS_CHOICE']
            del self.DEV['BIG_FC_INIT_LIST']
        if self.DEV.LOSS_CHOICE != 'ot':
            del self.DEV['OT_ONE_DIM_FORM']
        # if self.DEV.ASSIGN_BOX_ON_ALL_SCALE:
        #     del self.ROIS['ASSIGN_ANCHOR_BASE']


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific to the COCO dataset.
    """

    def __init__(self, args):
        super(CocoConfig, self).__init__()

        self.CTRL.CONFIG_NAME = args.config_name
        self.CTRL.PHASE = args.phase
        self.CTRL.DEBUG = args.debug

        self.MISC.DEVICE_ID = [int(x) for x in args.device_id.split(',')]
        self.MISC.GPU_COUNT = len(self.MISC.DEVICE_ID)

        _ignore_yaml = False
        # ================ (CUSTOMIZED CONFIG) =========================
        if args.config_name.startswith('local_pc') \
                or args.config_name.startswith('remote_debug_1'):

            # debug mode on local pc
            # self.CTRL.PROFILE_ANALYSIS = True
            self.MISC.USE_VISDOM = True
            self.MISC.VIS.PORT = 8097  # debug

            self.TRAIN.BATCH_SIZE = 2
            # self.TRAIN.INIT_LR = 0.005
            # self.DATA.IMAGE_MAX_DIM = 512
            # self.DATA.IMAGE_MIN_DIM = 512
            self.CTRL.QUICK_VERIFY = True

            self.DEV.SWITCH = True
            self.DEV.BUFFER_SIZE = 20
            self.DEV.LOSS_FAC = 50.
            self.DEV.LOSS_CHOICE = 'l2'
            self.DEV.OT_ONE_DIM_FORM = 'conv'  # 'fc'

            # self.DEV.DIS_REG_LOSS = True
            # self.ROIS.ASSIGN_ANCHOR_BASE = 40.  # useless when ASSIGN_BOX_ON_ALL_SCALE is True

            # self.DEV.STRUCTURE = 'alpha'
            self.DEV.BIG_SUPERVISE = False
            self.DEV.BIG_LOSS_FAC = .1
            self.DEV.BIG_FC_INIT = 'coco_pretrain'

            self.DEV.STRUCTURE = 'beta'
            self.DEV.DIS_UPSAMPLER = False
            self.DEV.UPSAMPLE_FAC = 1.0
            self.DEV.ASSIGN_BOX_ON_ALL_SCALE = False
            self.DEV.BIG_FEAT_DETACH = False
            self.DEV.INST_LOSS = True

            self.DEV.CLS_MERGE_FEAT = True
            self.DEV.CLS_MERGE_MANNER = 'simple_add'
            # self.DEV.CLS_MERGE_MANNER = 'linear_add'
            self.TRAIN.FPN_OT_LOSS = True
            self.TRAIN.FPN_OT_LOSS_FAC = .1

            self.ROIS.METHOD = 'roi_pool'

            # self.DEV.BASELINE = True  # apply up-sampling op. in original Mask-RCNN
            # self.DEV.MULTI_UPSAMPLER = False
            _ignore_yaml = True

        elif args.config_name.startswith('base_101'):
            self.MODEL.INIT_FILE_CHOICE = 'coco_pretrain'
            self.TRAIN.BATCH_SIZE = 16
            self.CTRL.PROFILE_ANALYSIS = False
            _ignore_yaml = True

        elif args.config_name.startswith('base_102'):
            self.MODEL.INIT_FILE_CHOICE = 'imagenet_pretrain'
            self.TRAIN.BATCH_SIZE = 16
            self.CTRL.PROFILE_ANALYSIS = False
            self.TEST.SAVE_IM = False
            _ignore_yaml = True

        elif args.config_name is None:
            if args.config_file is None:
                print('WARNING: No config file and config name! use default setting.'
                      'set config_name=default')
                self.CTRL.CONFIG_NAME = 'default'
            else:
                print('no config name but luckily you got config file ...')
        else:
            print('WARNING: unknown config name!!! use default setting.')
        # ================ (CUSTOMIZED CONFIG END) ======================

        # Optional (override previous config)
        if args.config_file is not None and not _ignore_yaml:
            print('Find .yaml file; use yaml name as CONFIG_NAME')
            self.CTRL.CONFIG_NAME = os.path.basename(args.config_file).replace('.yaml', '')
            merge_cfg_from_file(args.config_file, self)

        if len(args.opts) != 0:
            print('Update configuration from terminal inputs ...')
            merge_cfg_from_list(args.opts, self)

        self._set_value()
