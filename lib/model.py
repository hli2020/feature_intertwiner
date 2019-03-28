import re
from lib.sub_module import *
from lib.layers import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import tools.utils as utils
from lib.OT_module import OptTrans
from tools.image_utils import parse_image_meta


EPS = 1e-20


class MaskRCNN(nn.Module):
    def __init__(self, config):
        super(MaskRCNN, self).__init__()
        self.config = config
        self._build(config=config)
        self._initialize_weights()
    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, value):
        self._iter = value

    def _build(self, config):
        """Build Mask R-CNN architecture: fpn, rpn, classifier, mask"""
        # Image size must be dividable by 2 multiple times
        h, w = config.DATA.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the head (stage 5), so we pick the 4th item in the list.
        resnet = ResNet(config.MODEL.BACKBONE, stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()
        # Top-down Layers
        self.fpn = FPN(config, C1, C2, C3, C4, C5, out_channels=256)

        # Generate Anchors (Tensor; do not assign cuda() here)
        self.priors = torch.from_numpy(
            generate_pyramid_priors(config.RPN.ANCHOR_SCALES, config.RPN.ANCHOR_RATIOS,
                                    config.MODEL.BACKBONE_SHAPES, config.MODEL.BACKBONE_STRIDES,
                                    config.RPN.ANCHOR_STRIDE)).float()
        # RPN
        self.rpn = RPN(len(config.RPN.ANCHOR_RATIOS), config.RPN.ANCHOR_STRIDE, input_ch=256)
        # RoI
        self.dev_roi = Dev(config, depth=256)
        if self.config.DEV.LOSS_CHOICE == 'ot':
            self.ot_loss = OptTrans(self.config, ch_x=1024)

        # FPN Classifier
        self.classifier = \
            Classifier(depth=256, num_classes=config.DATASET.NUM_CLASSES,
                       pool_size=config.MRCNN.POOL_SIZE, config=config)
        # FPN Mask
        self.mask = Mask(depth=256, num_classes=config.DATASET.NUM_CLASSES)

        # Update (May 3): comment the following
        # if not config.TRAIN.BN_LEARN:
        #     # Fix batch norm layers
        #     def set_bn_fix(m):
        #         classname = m.__class__.__name__
        #         if classname.find('BatchNorm') != -1:
        #             for p in m.parameters():
        #                 p.requires_grad = False
        #     self.apply(set_bn_fix)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def initialize_buffer(self, log_file):
        """ called in 'utils.py' """
        if self.config.DEV.INIT_BUFFER_WEIGHT == 'scratch':
            utils.print_log('init buffer from scratch ...', log_file)
            self.buffer = torch.zeros(self.config.DEV.BUFFER_SIZE, 1024, self.config.DATASET.NUM_CLASSES).cuda()
            self.buffer_cnt = torch.zeros(self.config.DEV.BUFFER_SIZE, 1, self.config.DATASET.NUM_CLASSES).cuda()

        elif self.config.DEV.INIT_BUFFER_WEIGHT == 'coco_pretrain':
            utils.print_log('init buffer from pretrain model ...', log_file)
            NotImplementedError()

    def set_trainable(self, layer_regex, log_file):
        """called in 'workflow.py'
        Sets model layers as trainable if their names match the given regular expression.
        """
        # fpn.P5_conv1.bias
        # fpn.P5_conv2.1.weight
        # fpn.P5_conv2.1.bias
        # dev_roi.upsample.4.weight
        # dev_roi.upsample.4.bias
        # dev_roi.feat_extract.0.weight
        # dev_roi.feat_extract.0.bias
        # fpn.C5.0.bn3.weight
        # fpn.C5.0.bn3.bias
        # fpn.C5.0.downsample.0.weight
        # fpn.C5.0.downsample.0.bias
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True
        for name, param in self.named_parameters():
            utils.print_log('\tlayer name: {}\t\treguires_grad: {}'.format(name, param.requires_grad),
                      log_file, quiet_termi=True)

    def meta_loss(self, feat_input):
        """the loss is computed in GPU 0; called in workflow.py *only*."""
        # the direct outcome (feat_out) from 'forward() of Dev class in sub_module.py'
        [big_feat, big_cnt, small_feat, small_cnt, small_output_all, small_gt_all] = feat_input

        # update buffer (buffer_size x 1024 x 81)
        # self.buffer/buffer_cnt is Tensor
        buffer_size = self.buffer.size(0)
        _big_feat, _big_cnt = self._merge_feat_vec(big_feat, big_cnt)
        _big_feat_tensor, _big_cnt_tensor = _big_feat.data, _big_cnt.data
        if buffer_size == 1:
            # use all historic data
            feat_sum = self.buffer * self.buffer_cnt + _big_feat_tensor.unsqueeze(0) * _big_cnt_tensor.unsqueeze(0)
            self.buffer_cnt += _big_cnt_tensor.unsqueeze(0)
            self.buffer = feat_sum / (self.buffer_cnt + EPS)
            final_big_feat = self.buffer.squeeze()  # shape: 1024 x 81
        else:
            # in-place opt. on Tensor (cannot be done on Variable)
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1, :, :] = _big_feat_tensor
            self.buffer_cnt[:-1] = self.buffer_cnt[1:]
            self.buffer_cnt[-1, :, :] = _big_cnt_tensor
            final_big_feat = \
                torch.sum(self.buffer * self.buffer_cnt, dim=0) / (torch.sum(self.buffer_cnt, dim=0) + EPS)

        if self.config.DEV.INST_LOSS:
            # _idx_tmp/_idx indexes the instances of small objects
            # small_gt_all shape: 1200
            _idx_tmp = torch.nonzero(small_gt_all).squeeze().data
            buff_cls_idx = torch.nonzero(torch.sum(self.buffer_cnt, dim=0).squeeze() > 0).squeeze()
            _idx = [ind for ind in _idx_tmp if small_gt_all[ind].data.cpu().numpy() in buff_cls_idx]
            _idx = torch.from_numpy(np.array(_idx)).cuda()
        else:
            # final_small_feat, 1024 x 81; final_small_cnt, 1 x 81
            final_small_feat, final_small_cnt = self._merge_feat_vec(small_feat, small_cnt)
            final_small_cnt.data[0][0] = 0  # Variable; do not include background cls when computing meta_loss
            # _idx indexes the 81 classes
            _check = (final_small_cnt.squeeze() > 0) + (Variable(self.buffer_cnt.squeeze()) > 0)
            _idx = torch.nonzero(_check == 2).squeeze().data

        if _idx.size():
            if self.config.DEV.INST_LOSS:
                SMALL = small_output_all[_idx, :]
                BIG = self._assign_from_buffer(final_big_feat, small_gt_all[_idx])
            else:
                SMALL = final_small_feat[:, _idx].t()  # say 15 x 1024
                final_big_feat_var = Variable(final_big_feat)
                BIG = final_big_feat_var[:, _idx].t()   # also 15 x 1024
                # if self.config.CTRL.DEBUG:
                #     print('comp_cls_num in meta-loss: {}'.format(_idx.size(0)))
                #     print('SMALL mean: {:.4f}'.format(SMALL.mean().data.cpu()[0]))
                #     print('BIG mean: {:.4f}'.format(BIG.mean().data.cpu()[0]))

            # compute meta-loss
            if self.config.DEV.LOSS_CHOICE == 'l2':
                loss = F.mse_loss(SMALL, BIG)   # use sigmoid before comparison

            elif self.config.DEV.LOSS_CHOICE == 'kl':
                loss = F.kl_div(torch.log(SMALL), BIG)   # use softmax

            elif self.config.DEV.LOSS_CHOICE == 'l1':   # use sigmoid before comparison
                loss = F.l1_loss(SMALL, BIG)

            elif self.config.DEV.LOSS_CHOICE == 'ot':
                loss = self.ot_loss(SMALL.unsqueeze(dim=-1), BIG.unsqueeze(dim=-1).contiguous())
        else:
            loss = Variable(torch.zeros(1).cuda())
        return loss

    @staticmethod
    def _assign_from_buffer(buffer, list):
        out = torch.stack([buffer[:, cls] for cls in list.data.cpu().numpy()])
        return Variable(out)

    @staticmethod
    def _merge_feat_vec(box_feat, box_cnt):
        """merge [gpu_num, scale_num] into 1"""
        feat_avg_sum = box_feat * box_cnt
        feat_avg_sum = torch.sum(torch.sum(feat_avg_sum, dim=0), dim=0)  # 1024 x 81
        cnt_sum = torch.sum(torch.sum(box_cnt, dim=0), dim=0)  # 1 x 81
        feat_avg_sum /= (cnt_sum + EPS)
        return feat_avg_sum, cnt_sum

    @staticmethod
    def adjust_input_gt(*args):
        """zero-padding different number of GTs for each image within the batch"""
        gt_cls_ids = args[0]
        gt_boxes = args[1]
        gt_masks = args[2]
        gt_num = [x.shape[0] for x in gt_cls_ids]
        max_gt_num = max(gt_num)
        bs = len(gt_cls_ids)
        mask_shape = gt_masks[0].shape[1]

        GT_CLS_IDS = torch.zeros(bs, max_gt_num)
        GT_BOXES = torch.zeros(bs, max_gt_num, 4)
        GT_MASKS = torch.zeros(bs, max_gt_num, mask_shape, mask_shape)
        for i in range(bs):
            GT_CLS_IDS[i, :gt_num[i]] = torch.from_numpy(gt_cls_ids[i])
            GT_BOXES[i, :gt_num[i], :] = torch.from_numpy(gt_boxes[i]).float()
            GT_MASKS[i, :gt_num[i], :, :] = torch.from_numpy(gt_masks[i]).float()

        GT_CLS_IDS = Variable(GT_CLS_IDS.cuda(), requires_grad=False)
        GT_BOXES = Variable(GT_BOXES.cuda(), requires_grad=False)
        GT_MASKS = Variable(GT_MASKS.cuda(), requires_grad=False)

        return GT_CLS_IDS, GT_BOXES, GT_MASKS, gt_num

    def forward(self, input, mode, do_meta=False):
        """forward function of the Mask-RCNN network
            input: data
            mode: train, inference, visualize
            do_meta (not used for now):
                only affects the very few iterations during train under meta-loss case
        """
        molded_images = input[0]
        sample_per_gpu = molded_images.size(0)  # aka, actual batch size
        # for debug only
        curr_gpu_id = torch.cuda.current_device()
        curr_coco_im_id = input[-1][:, -1]

        # set model state
        if mode == 'inference' or 'visualize':
            _proposal_cnt = self.config.RPN.POST_NMS_ROIS_INFERENCE
            self.eval()
        elif mode == 'train':
            _proposal_cnt = self.config.RPN.POST_NMS_ROIS_TRAINING
            self.train()
            if not self.config.TRAIN.BN_LEARN:
                # Set BN layer always in eval mode during training
                def set_bn_eval(m):
                    classname = m.__class__.__name__
                    if classname.find('BatchNorm') != -1:
                        m.eval()
                self.apply(set_bn_eval)
        else:
            raise Exception('unknown phase')

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out, fpn_ot_loss] = self.fpn(molded_images, mode=mode)

        # Note that P6 is used in RPN, but not in the classifier heads.
        _rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        _mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in _rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate rpn layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_pred_cls_logits, _rpn_class_score, rpn_pred_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N (say 2000), (y1, x1, y2, x2)] in normalized coordinates and zero padded.
        _proposals = proposal_layer([_rpn_class_score, rpn_pred_bbox],
                                    proposal_count=_proposal_cnt,
                                    nms_threshold=self.config.RPN.NMS_THRESHOLD,
                                    priors=self.priors, config=self.config)
        # Normalize coordinates
        h, w = self.config.DATA.IMAGE_SHAPE[:2]
        scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False).cuda()

        if self.config.CTRL.PROFILE_ANALYSIS and mode == 'train':
            print('\t[gpu {:d}] curr_coco_im_ids: {}'.format(curr_gpu_id, curr_coco_im_id.data.cpu().numpy()))
            print('\t[gpu {:d}] pass feature extraction'.format(curr_gpu_id))

        if mode == 'inference':

            assert _proposals.sum().data[0] != 0

            _pooled_cls, _, _feat_out_test = self.dev_roi(_mrcnn_feature_maps, _proposals)

            if self.config.DEV.STRUCTURE == 'beta':
                small_output_all, small_gt_all = _feat_out_test
            else:
                small_output_all, small_gt_all = None, None

            _, mrcnn_class, mrcnn_bbox = self.classifier(_pooled_cls, small_output_all, small_gt_all)

            # Detections
            # input[1], image_metas, (3, 90), Variable
            _, _, windows, _, _ = parse_image_meta(input[1])
            # output is [batch, num_detections (say 100), (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(_proposals, mrcnn_class, mrcnn_bbox, windows, self.config)

            # assert detections.sum().data[0] != 0   # update: allow zero detection
            # Convert boxes to normalized coordinates
            normalize_boxes = detections[:, :, :4] / scale
            # Create masks for detections
            _, _pooled_mask, _ = self.dev_roi(_mrcnn_feature_maps, normalize_boxes)
            mrcnn_mask = self.mask(_pooled_mask)

            # shape: batch, num_detections, 81, 28, 28
            mrcnn_mask = mrcnn_mask.view(
                sample_per_gpu, -1, mrcnn_mask.size(1), mrcnn_mask.size(2), mrcnn_mask.size(3))

            return [detections, mrcnn_mask]

        elif mode == 'visualize':

            assert _proposals.sum().data[0] != 0

            _pooled_cls, _, _feat_out_test = self.dev_roi(_mrcnn_feature_maps, _proposals)

            if self.config.DEV.STRUCTURE == 'beta':
                small_output_all, small_gt_all = _feat_out_test
            else:
                small_output_all, small_gt_all = None, None

            feature, mrcnn_class, mrcnn_bbox = self.classifier(_pooled_cls, small_output_all, small_gt_all)

            # Detections
            # input[1], image_metas, (3, 90), Variable
            _, _, windows, _, _ = parse_image_meta(input[1])
            # output is [batch, num_detections (say 100), (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections, out_feat = detection_layer(_proposals, mrcnn_class, mrcnn_bbox, windows, self.config,
                                                   feature, small_gt_all)
            # NO MASK BRANCH
            return [detections, out_feat]

        elif mode == 'train':

            gt_class_ids, gt_boxes, gt_masks = input[1], input[2], input[3]

            # 1. compute RPN targets
            # try:
            target_rpn_match, target_rpn_bbox = \
                prepare_rpn_target(self.priors, gt_class_ids, gt_boxes, self.config, curr_coco_im_id)
            # except RuntimeError:
            #     import pdb
            #     pdb.set_trace()
            #     a = 1
 
            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass rpn_target generation'.format(curr_gpu_id))

            # 2. compute DET targets
            # _rois shape: bs x TRAIN_ROIS_PER_IMAGE (say 200) x 4; zero padded
            # target_class_ids: bs, 200
            _rois, target_class_ids, target_deltas, target_mask = \
                prepare_det_target(_proposals.detach(), gt_class_ids, gt_boxes / scale, gt_masks, self.config)
            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass pass det_target generation'.format(curr_gpu_id))

            # 3.0 preview: initialize output
            # big_feat/small_feat shape: gpu_num, scale_num, feat_dim, cls_num; used for meta-loss
            scale_num = 2 if self.config.DEV.STRUCTURE == 'alpha' else 3
            if self.config.DEV.ASSIGN_BOX_ON_ALL_SCALE:
                scale_num = 4
            big_feat = Variable(torch.zeros(1, scale_num, 1024, self.config.DATASET.NUM_CLASSES).cuda())
            big_cnt = Variable(torch.zeros(1, scale_num, 1, self.config.DATASET.NUM_CLASSES).cuda())
            small_feat = Variable(torch.zeros(1, scale_num, 1024, self.config.DATASET.NUM_CLASSES).cuda())
            small_cnt = Variable(torch.zeros(1, scale_num, 1, self.config.DATASET.NUM_CLASSES).cuda())
            big_loss = Variable(torch.zeros(1, scale_num, 1).cuda())

            small_output_all = Variable(torch.zeros(1, 1024).cuda())
            small_gt_all = Variable(torch.zeros(1).cuda())

            num_rois, mask_sz, num_cls = \
                self.config.ROIS.TRAIN_ROIS_PER_IMAGE, self.config.MRCNN.MASK_SHAPE[0], self.config.DATASET.NUM_CLASSES
            mrcnn_class_logits = Variable(torch.zeros(sample_per_gpu, num_rois, num_cls).cuda())
            mrcnn_bbox = Variable(torch.zeros(sample_per_gpu, num_rois, num_cls, 4).cuda())
            mrcnn_mask = Variable(torch.zeros(sample_per_gpu, num_rois, num_cls, mask_sz, mask_sz).cuda())

            # 3. mask and cls generation
            if torch.sum(_rois).data[0] != 0:
                # COMPUTE META_OUTPUTS HERE
                # _pooled_cls: 600 (bsx200), 256, 7, 7
                _pooled_cls, _pooled_mask, _feat_out = \
                    self.dev_roi(_mrcnn_feature_maps, _rois, target_class_ids)

                if self.config.DEV.SWITCH and not self.config.DEV.BASELINE:
                    if self.config.DEV.STRUCTURE == 'beta':
                        [big_feat, big_cnt, small_feat, small_cnt, big_loss,
                         small_output_all, small_gt_all] = _feat_out
                    elif self.config.DEV.STRUCTURE == 'alpha':
                        [big_feat, big_cnt, small_feat, small_cnt, big_loss] = _feat_out
                    # if self.config.DEV.ASSIGN_BOX_ON_ALL_SCALE:
                    #     assert big_feat.size() == (1, 4, 1024, 81), 'big_feat size: {}'.format(big_feat.size())
                    #     assert small_feat.size() == big_feat.size(), 'small_feat size: {}'.format(small_feat.size())
                    #     assert big_cnt.size() == (1, 4, 1, 81), 'big_cnt size: {}'.format(big_cnt.size())
                    #     assert small_cnt.size() == big_cnt.size(), 'small_cnt size: {}'.format(small_cnt.size())
                    # if self.config.CTRL.DEBUG:
                    #     print('pooled_cls mean: {:.4f}'.format(_pooled_cls.mean().data.cpu()[0]))
                    #     print('pooled_mask mean: {:.4f}'.format(_pooled_mask.mean().data.cpu()[0]))
                # classifier
                mrcnn_class_logits, _, mrcnn_bbox = \
                    self.classifier(_pooled_cls, small_output_all, small_gt_all)
                # if self.config.CTRL.DEBUG:
                #     a, b = torch.max(mrcnn_cls_logits, dim=-1)
                #     print('train classifier, ROIs pred_cls sum: {}'.format(b.sum().data[0]))

                # mask
                mrcnn_mask = self.mask(_pooled_mask)

                # reshape output
                mrcnn_class_logits = mrcnn_class_logits.view(sample_per_gpu, -1, mrcnn_class_logits.size(1))
                mrcnn_bbox = mrcnn_bbox.view(sample_per_gpu, -1, mrcnn_bbox.size(1), mrcnn_bbox.size(2))
                mrcnn_mask = mrcnn_mask.view(
                    sample_per_gpu, -1, mrcnn_mask.size(1), mrcnn_mask.size(2), mrcnn_mask.size(3))

            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass mask and cls generation'.format(curr_gpu_id))

            # 4. compute loss
            rpn_class_loss = compute_rpn_class_loss(target_rpn_match, rpn_pred_cls_logits)
            rpn_bbox_loss = compute_rpn_bbox_loss(target_rpn_bbox, target_rpn_match, rpn_pred_bbox)
            mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
            mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
            mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

            loss_merge = torch.stack(
                (rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss), dim=1)
            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass loss compute!'.format(curr_gpu_id))

            # must be Variables
            return loss_merge, \
                   big_feat, big_cnt, small_feat, small_cnt, big_loss, \
                   small_output_all, small_gt_all, \
                   fpn_ot_loss
        # END TRAIN

