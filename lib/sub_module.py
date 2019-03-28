from lib.layers import pyramid_roi_align
from lib.roi_align.crop_and_resize import CropAndResizeFunction
from lib.roi_pooling.functions.roi_pool import RoIPoolFunction
import torch.nn.functional as F
from tools.utils import *
from .OT_module import OptTrans


class SamePad2d(nn.Module):
    """Mimic tensorflow's 'SAME' padding."""
    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


############################################################
#  Resnet Graph
############################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


############################################################
#  FPN Graph
############################################################
# not used
# class TopDownLayer(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(TopDownLayer, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#         self.padding2 = SamePad2d(kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
#
#     def forward(self, x, y):
#         y = F.upsample(y, scale_factor=2)
#         x = self.conv1(x)
#         return self.conv2(self.padding2(x+y))
class FPN(nn.Module):
    def __init__(self, config, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.config = config
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

        if self.config.TRAIN.FPN_OT_LOSS:
            self.ot = True
            base_size = int(self.config.DATA.IMAGE_SHAPE[0] / 4)
            self.p2_ot = OptTrans(config, ch_x=256, spatial_x=base_size/2, spatial_y=base_size)
            self.p3_ot = OptTrans(config, ch_x=256, spatial_x=base_size/4, spatial_y=base_size/2)
            self.p4_ot = OptTrans(config, ch_x=256, spatial_x=base_size/8, spatial_y=base_size/4)
            # self.p5_ot = OptTrans(config, ch_x=256, spatial_x=base_size/16, spatial_y=base_size/8)
        else:
            self.ot = False

    def forward(self, x, mode):
        bs = x.size(0)
        ot_loss = Variable(torch.zeros(bs, 3).cuda())
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)

        if self.ot and mode == 'train':
            tmp = self.P4_conv1(c4_out)
            ot_loss[:, 0] = self.p4_ot(p5_out, tmp)
            p4_out = tmp + F.upsample(p5_out, scale_factor=2)

            tmp = self.P3_conv1(c3_out)
            ot_loss[:, 1] = self.p3_ot(p4_out, tmp)
            p3_out = tmp + F.upsample(p4_out, scale_factor=2)

            tmp = self.P2_conv1(c2_out)
            ot_loss[:, 2] = self.p2_ot(p3_out, tmp)
            p2_out = tmp + F.upsample(p3_out, scale_factor=2)
        else:
            p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
            p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
            p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out, ot_loss]


############################################################
#  Region Proposal Network
############################################################
class RPN(nn.Module):
    """Builds the model of Region Proposal Network.
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    def __init__(self, anchors_per_location, anchor_stride, input_ch):
        super(RPN, self).__init__()
        self.anchor_stride = anchor_stride
        self.input_ch = input_ch

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.input_ch, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  DEV
############################################################
class Dev(nn.Module):
    def __init__(self, config, depth):
        super(Dev, self).__init__()
        self.depth = depth
        self.use_dev = config.DEV.SWITCH
        self.pool_size = config.MRCNN.POOL_SIZE
        self.mask_pool_size = config.MRCNN.MASK_POOL_SIZE
        self.image_shape = config.DATA.IMAGE_SHAPE
        self.num_classs = config.DATASET.NUM_CLASSES
        self.config = config
        self.dis_upsample = config.DEV.DIS_UPSAMPLER
        self.structure = config.DEV.STRUCTURE
        self.roi_type = config.ROIS.METHOD
        # if self.roi_type == 'roi_pool':
        self.roi_spatial_scale = [1./4, 1./8, 1./16, 1./32]

        if self.use_dev:
            # for now it's the same size with mask_pool_size (14)
            self.feat_pool_size = config.DEV.FEAT_BRANCH_POOL_SIZE
            self.upsample_fac = config.DEV.UPSAMPLE_FAC
            assert self.feat_pool_size % 2 == 0, 'pool size of feature branch has to be even'

            # define **upsampler**
            if not self.dis_upsample:
                if self.config.DEV.UPSAMPLE_FAC == 1.:
                    conv_opt = nn.Conv2d(self.depth, self.depth, kernel_size=3, padding=1)
                elif self.config.DEV.UPSAMPLE_FAC == 2.:
                    conv_opt = nn.ConvTranspose2d(self.depth, self.depth, kernel_size=3,
                                                  stride=2, padding=1, output_padding=1)
                upsample_num = 4 if self.config.DEV.MULTI_UPSAMPLER else 1

                # the make-up layer in the paper
                self.upsample = nn.ModuleList()
                for i in range(upsample_num):
                    self.upsample.append(nn.Sequential(*[
                        conv_opt,
                        nn.BatchNorm2d(self.depth),
                        nn.ReLU(inplace=True),
                        # nn.Conv2d(self.depth, self.depth, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(self.depth),
                        # nn.ReLU(inplace=True)
                    ]))

            # define **feature extractor** to be compared
            if not self.config.DEV.BASELINE:
                _ksize = int(self.feat_pool_size / 2)
                _layer_list = [
                    nn.Conv2d(self.depth, 512, kernel_size=3, padding=1, stride=2),   # halve the map
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 1024, kernel_size=_ksize, stride=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True)
                ]
                # the critic module in the feature intertwiner
                # shape: say 40, 1024, 1, 1
                self.feat_extract = nn.Sequential(*_layer_list)
                # define last_op, for ot, there is none
                if config.DEV.LOSS_CHOICE == 'l2' or config.DEV.LOSS_CHOICE == 'l1':
                    self.last_op = nn.Sigmoid()
                elif config.DEV.LOSS_CHOICE == 'kl':
                    self.last_op = nn.Softmax(dim=1)

                if self.config.DEV.BIG_SUPERVISE:
                    self.big_fc_layer = nn.Linear(1024, self.num_classs)

    @staticmethod
    def _find_big_box(level, roi_level):
        if level == 2:
            big_ix = (roi_level == 4) + (roi_level == 5)
        elif level == 3:
            big_ix = (roi_level == 5)
        else:
            # for now, higher scale 4, 5 do not apply meta_loss
            big_ix = (roi_level == -1)
        return big_ix

    @staticmethod
    def _find_big_box2(level, roi_level):
        """called in structure='beta' or above"""
        if level == 2:
            big_ix = (roi_level == 3) + (roi_level == 4) + (roi_level == 5)
            # big_ix = (roi_level == 4) + (roi_level == 5)
        elif level == 3:
            big_ix = (roi_level == 4) + (roi_level == 5)
        elif level == 4:
            big_ix = (roi_level == 5)
        elif level == 5:
            big_ix = (roi_level == -1)
        return big_ix

    def forward(self, x, rois, roi_cls_gt=None):
        # x is a multi-scale List containing Variable (feature maps)
        # rois: [bs, 200, 4], normalized, y1, x1, y2, x2
        base = self.config.ROIS.ASSIGN_ANCHOR_BASE

        if not self.use_dev:
            # in 'layers.py'
            pooled_out = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape, base=base)
            mask_out = pyramid_roi_align([rois] + x, self.mask_pool_size, self.image_shape, base=base)
            feat_out = None

        elif self.structure == 'beta':
            SHOW_STAT = False
            # used for splitting train and test
            train_phase = False if roi_cls_gt is None else True

            # Step 1. Assign each ROI to a level in the pyramid based on the ROI area.
            y1, x1, y2, x2 = rois.chunk(4, dim=2)   # in normalized coordinate
            h, w = y2 - y1, x2 - x1
            area = w*h
            total_box = rois.size(0)*rois.size(1)

            # use **either** 'roi_level' or 'accu_small_idx' to assign anchors
            if not self.config.DEV.ASSIGN_BOX_ON_ALL_SCALE:
                # original plan
                _image_area = Variable(torch.FloatTensor(
                    [float(self.image_shape[0]*self.image_shape[1])]), requires_grad=False).cuda()
                roi_level = 4 + log2(torch.sqrt(area)/(base/torch.sqrt(_image_area)))
                roi_level = roi_level.round().int()
                # in case batch size =1, we keep that dim
                roi_level = roi_level.clamp(2, 5).squeeze(dim=-1)   # size: [bs, num_roi], say [3, 200]
            else:
                accu_small_idx = Variable(torch.ByteTensor(rois.size(0), rois.size(1)).cuda())
                accu_small_idx[:] = False

            if SHOW_STAT:
                print('\tassign ROIs (total num: {:d}) in {:d} scales.'
                      'max box area: {:.4f}, min box area: {:.4f}'.format(
                        total_box, 4, area.max().data[0], area.min().data[0]))

            # Step 2. LOOP through levels and apply ROI pooling to each.
            # P2 to P5, with 2 being the most coarse map
            pooled, mask, box_to_level = [], [], []
            big_feat, big_cnt, small_feat, small_cnt = [], [], [], []   # to generate feat_out
            big_loss = []
            small_output_all = Variable(torch.zeros(total_box, 1024).cuda())
            small_gt_all = Variable(torch.zeros(total_box).cuda())
            small_out_cnt = 0

            for i, level in enumerate(range(2, 6)):

                curr_feat_maps = x[i]

                # decide if use meta-loss on current scale
                if not self.config.DEV.ASSIGN_BOX_ON_ALL_SCALE:
                    _use_meta = True if level in [2, 3, 4] else False
                else:
                    _use_meta = True

                # Decide "small_ix": bs, num_roi
                _thres = (self.feat_pool_size / curr_feat_maps.size(-1))**2
                if not self.config.DEV.ASSIGN_BOX_ON_ALL_SCALE:
                    small_ix = roi_level == level
                else:
                    # new plan: these boxes is smaller than RoI output
                    # Note: on the last scale (5), there might have some big boxes; deem it as supervision (target).
                    _temp = area.squeeze(-1) <= _thres
                    small_ix = _temp - accu_small_idx
                    accu_small_idx = _temp

                # for inference, merge the big box index with small on the last scale
                if self.config.DEV.ASSIGN_BOX_ON_ALL_SCALE and not train_phase and level == 5:
                    # can't use big_ix during inference (since it's not generated)
                    # small_ix = (small_ix + big_ix) > 0
                    small_ix = ((accu_small_idx == 0) + small_ix) > 0

                if not small_ix.any():
                    if SHOW_STAT:
                        print('\tscale {:d} (thres: {:.4f}), NO (small) num_box, skip this scale ...'
                              .format(level, _thres))
                    # if there are no "small" boxes, we won't compute stats of *both* small and big on this scale
                    if _use_meta and train_phase and not self.config.DEV.BASELINE:
                        small_feat.append(Variable(torch.zeros(1024, self.num_classs).cuda()))
                        small_cnt.append(Variable(torch.zeros(1, self.num_classs).cuda(), requires_grad=False))
                        big_feat.append(Variable(torch.zeros(1024, self.num_classs).cuda()))
                        big_cnt.append(Variable(torch.zeros(1, self.num_classs).cuda(), requires_grad=False))
                        big_loss.append(Variable(torch.zeros(1).cuda()))
                    continue
                #import pdb 
                #pdb.set_trace()

                # TRAIN ONLY: Decide "big_ix"; deal with 'big' boxes
                if train_phase and not self.config.DEV.BASELINE:
                    if not self.config.DEV.ASSIGN_BOX_ON_ALL_SCALE:
                        big_ix = self._find_big_box2(level, roi_level)
                    else:
                        big_ix = accu_small_idx == 0

                    if not big_ix.any():
                        if _use_meta:
                            # there is no "big" boxes; never mind, we use historic data
                            big_feat.append(Variable(torch.zeros(1024, self.num_classs).cuda()))
                            big_cnt.append(Variable(torch.zeros(1, self.num_classs).cuda(), requires_grad=False))
                            big_loss.append(Variable(torch.zeros(1).cuda()))
                        big_num = 0
                        big_no_need = 0
                        big_need = 0
                    else:
                        # process big-small-supervise (big part)
                        big_index = torch.nonzero(big_ix)
                        big_boxes = rois[big_index[:, 0].data, big_index[:, 1].data, :]
                        big_box_gt = roi_cls_gt[big_index[:, 0].data, big_index[:, 1].data]
                        # for big boxes, ROI-pool on original map
                        big_box_ind = big_index[:, 0].int()

                        # NOT working below
                        # _idx = i if self.config.DEV.MULTI_UPSAMPLER else 0
                        # _feat_maps = self.upsample[_idx](curr_feat_maps)
                        _feat_maps = curr_feat_maps
                        # big_feat_pooled shape: say 20, 256, 14, 14
                        if self.roi_type == 'roi_align':
                            big_feat_pooled = CropAndResizeFunction(
                                self.feat_pool_size, self.feat_pool_size)(_feat_maps, big_boxes, big_box_ind)
                        elif self.roi_type == 'roi_pool':
                            new_input = self._make_roi_pool_box_input(big_boxes, big_box_ind)
                            big_feat_pooled = RoIPoolFunction(
                                self.feat_pool_size, self.feat_pool_size, self.roi_spatial_scale[i]
                            )(_feat_maps, new_input)

                        # shape: say 20, 1024, 1, 1
                        _big_out_before_last = self.feat_extract(big_feat_pooled)
                        if self.config.DEV.LOSS_CHOICE != 'ot':
                            big_output = self.last_op(_big_out_before_last)
                        else:
                            big_output = _big_out_before_last

                        # transfer ins2cls feature
                        # shape: always 1024 x 81 (cls_num); this is an averaged output
                        _b_feat, _b_cnt = self._assign_feat2cls([big_box_gt, big_output])

                        big_feat.append(_b_feat)
                        big_cnt.append(_b_cnt)
                        big_num = big_index.size(0)
                        if SHOW_STAT:
                            big_area = (big_boxes[:, 0] - big_boxes[:, 2])*(big_boxes[:, 1] - big_boxes[:, 3])
                            big_no_need = torch.nonzero(big_area >= _thres).size(0)
                            try:
                                big_need = torch.nonzero(big_area < _thres).size(0)
                            except RuntimeError:
                                big_need = 0

                        if self.config.DEV.BIG_SUPERVISE:
                            big_x = _big_out_before_last.view(-1, 1024)
                            big_feat_cls_digits = self.big_fc_layer(big_x)   # big_num x 81
                            curr_big_loss = F.cross_entropy(big_feat_cls_digits, big_box_gt.long())
                            big_loss.append(curr_big_loss)
                        else:
                            big_loss.append(Variable(torch.zeros(1).cuda()))

                # "SMALL" boxes (or simply boxes on scale 4,5) exist
                # small_index: say, 2670 (actual boxes found in this level) x 2
                small_index = torch.nonzero(small_ix)
                # Keep track of which box is mapped to which level
                box_to_level.append(small_index.data)
                # rois: [bs, num_roi, 4] -> small_boxes [index[0], 4]
                small_boxes = rois[small_index[:, 0].data, small_index[:, 1].data, :]

                # scale up feature map of "smaller" boxes
                box_ind = small_index[:, 0].int()
                _idx = i if self.config.DEV.MULTI_UPSAMPLER else 0
                _feat_maps = self.upsample[_idx](curr_feat_maps)
                # _feat_maps = curr_feat_maps
                assert small_boxes.max().data[0] <= 1.0

                # pooled_features shape: say 473, 256, 7, 7
                if self.roi_type == 'roi_align':
                    pooled_features = CropAndResizeFunction(
                        self.pool_size, self.pool_size)(_feat_maps, small_boxes, box_ind)
                elif self.roi_type == 'roi_pool':
                    _input = self._make_roi_pool_box_input(small_boxes, box_ind)
                    pooled_features = RoIPoolFunction(
                        self.pool_size, self.pool_size, self.roi_spatial_scale[i]
                    )(_feat_maps, _input)
                # print(pooled_features_align.sum().data[0])
                # print(pooled_features_pool.sum().data[0])
                pooled.append(pooled_features)

                # mask and feat features are shared with a RoI
                # since the output size is the same (mask_pool_size=feat_pool_size)
                # mask_and_feat shape: say 473, 256, 14, 14
                if self.roi_type == 'roi_align':
                    mask_and_feat = CropAndResizeFunction(
                        self.mask_pool_size, self.mask_pool_size)(_feat_maps, small_boxes, box_ind)
                elif self.roi_type == 'roi_pool':
                    mask_and_feat = RoIPoolFunction(
                        self.mask_pool_size, self.mask_pool_size, self.roi_spatial_scale[i]
                    )(_feat_maps, _input)
                mask.append(mask_and_feat)

                # Deal with 'small' boxes during train and test
                if _use_meta and not self.config.DEV.BASELINE:
                    # shape: say 460, 1024, 1, 1
                    small_output = self.feat_extract(mask_and_feat)
                    if self.config.DEV.LOSS_CHOICE != 'ot':
                        small_output = self.last_op(small_output)

                    _start_ind = small_out_cnt
                    _small_num = small_index.size(0)
                    small_output_all[_start_ind:_small_num+_start_ind, :] = small_output

                    if train_phase:
                        small_box_gt = roi_cls_gt[small_index[:, 0].data, small_index[:, 1].data]
                        # shape: always 1024 x 81 (cls_num); this is an averaged output
                        _s_feat, _s_cnt = self._assign_feat2cls([small_box_gt, small_output])
                        small_feat.append(_s_feat)
                        small_cnt.append(_s_cnt)
                        small_gt_all[_start_ind:_small_num+_start_ind] = small_box_gt
                    else:
                        small_gt_all[_start_ind:_small_num+_start_ind] = 1

                    small_out_cnt += _small_num

                if SHOW_STAT:
                    small_area = (small_boxes[:, 0] - small_boxes[:, 2])*(small_boxes[:, 1] - small_boxes[:, 3])
                    try:
                        no_need = torch.nonzero(small_area >= _thres).size(0)
                    except RuntimeError:
                        no_need = 0
                    try:
                        small_need = torch.nonzero(small_area < _thres).size(0)
                    except RuntimeError:
                        small_need = 0
                    print('\tscale {:d} (thres: {:.4f}), (small) num_box: {:d}, big_box: {:d}, meta_loss: {}\n\t\t'
                          'larger than thres: {:d}, smaller than thres {:d}\t\tFor big: larger: {:d}, smaller {:d}'
                          .format(level, _thres, small_index.size(0), big_num, _use_meta,
                                  no_need, small_need, big_no_need, big_need))
                    a = 1
            # SCALE LOOP ENDS

            pooled_out, mask_out = self._reshape_result(pooled, mask, box_to_level, rois.size())

            if train_phase and not self.config.DEV.BASELINE:
                if self.config.DEV.BIG_FEAT_DETACH:
                    # do *NOT* pass gradient of big_feat
                    big_feat = torch.stack(big_feat).unsqueeze(dim=0).detach()
                else:
                    big_feat = torch.stack(big_feat).unsqueeze(dim=0)
                feat_out = [
                    big_feat,
                    torch.stack(big_cnt).unsqueeze(dim=0),
                    torch.stack(small_feat).unsqueeze(dim=0),
                    torch.stack(small_cnt).unsqueeze(dim=0),
                    torch.stack(big_loss).unsqueeze(dim=0),
                    small_output_all,
                    small_gt_all
                ]
            elif not train_phase:
                # test phase in 'beta' structure
                feat_out = [small_output_all, small_gt_all]
            else:
                feat_out = []
        # END BETA STRUCTURE
        return pooled_out, mask_out, feat_out

    @staticmethod
    def _reshape_result(pooled, mask, box_to_level, rois_size):
        pooled = torch.cat(pooled, dim=0)
        mask = torch.cat(mask, dim=0)
        box_to_level = torch.cat(box_to_level, dim=0)

        # Rearrange pooled features to match the order of the original boxes
        pooled_out = Variable(torch.zeros(
            rois_size[0], rois_size[1], pooled.size(1), pooled.size(2), pooled.size(3)).cuda())
        pooled_out[box_to_level[:, 0], box_to_level[:, 1], :, :, :] = pooled
        # 3, 1000, 256, 7, 7 -> 3000, 256, 7, 7
        pooled_out = pooled_out.view(-1, pooled_out.size(2), pooled_out.size(3), pooled_out.size(4))

        mask_out = Variable(torch.zeros(
            rois_size[0], rois_size[1], mask.size(1), mask.size(2), mask.size(3)).cuda())
        mask_out[box_to_level[:, 0], box_to_level[:, 1], :, :, :] = mask
        mask_out = mask_out.view(-1, mask_out.size(2), mask_out.size(3), mask_out.size(4))

        return pooled_out, mask_out

    def _assign_feat2cls(self, input):
        """
        input[List]
                each entry: Variable; [20], [20, 1024, 1, 1], [473], [473, 1024] on scale 2
                            Variable; [9], [9, 1024], [56], [56, 1024] on scale 3
        """
        box_gt, input_feat = input[0], input[1]
        assert box_gt.size(0) == input_feat.size(0)

        feat = Variable(torch.zeros(1024, self.num_classs).cuda())
        cnt = Variable(torch.zeros(1, self.num_classs).cuda(), requires_grad=False)

        for cls_ind in unique1d(box_gt).data:
            if cls_ind == 0:
                continue  # skip background
            else:
                _idx = torch.nonzero(box_gt == cls_ind).squeeze()
                cnt[0, cls_ind] = _idx.size(0)
                # left: 1024 x 81; right result: 1024 x 1 x 1; no need to squeeze the right result
                feat[:, cls_ind] = torch.mean(input_feat[_idx, :], dim=0)
        return feat, cnt

    def _make_roi_pool_box_input(self, boxes, box_ind):
        # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
        boxes_new = boxes * float(self.image_shape[0])
        # boxes *= float(self.image_shape[0])
        _y1, _x1, _y2, _x2 = boxes_new.chunk(4, dim=1)
        new_input = torch.stack([box_ind.float().unsqueeze(1), _x1, _y1, _x2, _y2], dim=1).squeeze(dim=-1)
        return new_input


############################################################
#  Feature Pyramid Network Heads
############################################################
class Classifier(nn.Module):
    def __init__(self, depth, num_classes, pool_size, config):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.config = config
        self.merge_meta = config.DEV.CLS_MERGE_FEAT

        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)

        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)

        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, small_feat_input, small_gt_index, mode='train'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.config.DEV.SWITCH and self.merge_meta and self.config.DEV.STRUCTURE == 'beta':
            if self.config.DEV.CLS_MERGE_MANNER == 'simple_add':
                x += (small_feat_input*(small_gt_index > 0).float().unsqueeze(1)).view(x.size(0), x.size(1), 1, 1)
            elif self.config.DEV.CLS_MERGE_MANNER == 'linear_add':
                _weights = (small_gt_index > 0).float() * self.config.DEV.CLS_MERGE_FAC
                _weights = _weights.view(x.size(0), 1, 1, 1)
                x = (1-_weights)*x + _weights*small_feat_input.view(x.size(0), x.size(1), 1, 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)           # x shape: bs x rois_num, 1024; used for CE loss
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size(0), -1, 4)

        if self.config.CTRL.PHASE == 'visualize':
            return [x, mrcnn_probs, mrcnn_bbox]
        else:
            # for train and inference
            return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):
    def __init__(self, depth, num_classes):
        super(Mask, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        # output is 28 x 28; matches the mask_shape
        return x
