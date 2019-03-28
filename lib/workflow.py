import matplotlib.pyplot as plt
from datasets.eval.PythonAPI.pycocotools import mask as maskUtils
from datasets.eval.PythonAPI.pycocotools.cocoeval import COCOeval
from tools.visualize import display_instances
from tools.image_utils import *
from tools.utils import *
import torch.nn as nn
from lib.config import LAYER_REGEX, TEMP, CLASS_NAMES

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse


def train_model(input_model, train_generator, valset, optimizer, layers, vis=None, coco_api=None):
    """
    Args:
        input_model:        nn.DataParallel
        train_generator:    Dataloader
        valset:             Dataset
        optimizer:          The learning rate to train with
        layers:
                            (only valid when END2END=False)
                            Allows selecting wich layers to train. It can be:
                                - A regular expression to match layer names to train
                                - One of these predefined values:
                                heads: The RPN, classifier and mask heads of the network
                                all: All the layers
                                3+: Train Resnet stage 3 and up
                                4+: Train Resnet stage 4 and up
                                5+: Train Resnet stage 5 and up
        vis:
        coco_api:            validation api
    """
    stage_name = layers.upper()
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    num_train_im = train_generator.dataset.dataset.num_images
    iter_per_epoch = math.floor(num_train_im/model.config.TRAIN.BATCH_SIZE)
    total_ep_till_now = sum(model.config.TRAIN.SCHEDULE[:TEMP[layers]])

    # check details
    if (num_train_im % model.config.TRAIN.BATCH_SIZE) % model.config.MISC.GPU_COUNT != 0:
        print_log('WARNING [TRAIN]: last mini-batch in an epoch is not divisible by gpu number.\n'
                  'total train im: {:d}, batch size: {:d}, gpu num {:d}\n'
                  'last mini-batch size: {:d}\n'.format(
                    num_train_im, model.config.TRAIN.BATCH_SIZE, model.config.MISC.GPU_COUNT,
                    (num_train_im % model.config.TRAIN.BATCH_SIZE)), model.config.MISC.LOG_FILE)

    if model.epoch > total_ep_till_now:
        print_log('skip {:s} stage ...'.format(stage_name.upper()), model.config.MISC.LOG_FILE)
        return None

    print_log('\n[Current stage: {:s}] start training at epoch {:d}, iter {:d}. \n'
              'Total epoch in this stage: {:d}.'.format(stage_name,
                model.epoch, model.iter, model.config.TRAIN.SCHEDULE[TEMP[layers]-1]),
                model.config.MISC.LOG_FILE)

    # Do NOT forget to modify LAYER_REGEX when adding new layers!
    if not model.config.TRAIN.END2END:
        if layers in LAYER_REGEX.keys():
            regx = LAYER_REGEX[layers]
            print_log('Stage: [{}] setting some layers trainable or not; '
                      'detail shown in log file; NOT shown in terminal'.format(stage_name), model.config.MISC.LOG_FILE)
            model.set_trainable(regx, model.config.MISC.LOG_FILE)
        else:
            raise Exception('unknown layer choice')

    # EPOCH LOOP
    for ep in range(model.epoch, total_ep_till_now+1):

        epoch_str = "[Ep {:03d}/{}]".format(ep, total_ep_till_now)
        print_log(epoch_str, model.config.MISC.LOG_FILE)
        # Training
        loss_data = train_epoch(input_model, train_generator, optimizer,
                                stage_name=stage_name, epoch_str=epoch_str,
                                epoch=ep, start_iter=model.iter, total_iter=iter_per_epoch,
                                valset=valset, coco_api=coco_api, vis=vis)
        # save model
        print_log('\n**Epoch ends**', model.config.MISC.LOG_FILE)
        info_pass = {
            'epoch':        ep,                 # or model.epoch
            'iter':         iter_per_epoch,     # or model.iter
            'loss_data':    loss_data
        }
        save_model(model, **info_pass)

        # one epoch ends; update iterator
        model.iter = 1
        model.epoch = ep

    # Current stage ends; do validation if possible
    model.epoch += 1
    # TODO(low): delete redundant model files to save hard-drive space
    if model.config.TRAIN.DO_VALIDATION:
        print_log('\nDo validation at end of current stage [{:s}] (model ep {:d} iter {:d}) ...'.
                  format(stage_name.upper(), total_ep_till_now, iter_per_epoch), model.config.MISC.LOG_FILE)
        test_model(input_model, valset, coco_api, during_train=True, epoch=ep, iter=iter_per_epoch, vis=vis)


def train_epoch(input_model, data_loader, optimizer, **args):
    """new training flow scheme
    Args:
        input_model
        data_loader
        optimizer
    """
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model
    config = model.config
    vis = args['vis']

    start_iter, total_iter, curr_ep = args['start_iter'], args['total_iter'], args['epoch']
    actual_total_iter = total_iter - start_iter + 1
    save_iter_base = math.floor(total_iter / config.TRAIN.SAVE_FREQ_WITHIN_EPOCH)

    # create iterator (deprecated)
    # data_iterator = iter(data_loader)
    if curr_ep == 1 and config.DEV.SWITCH:
        do_meta_after_iter = math.floor(config.DEV.EFFECT_AFER_EP_PERCENT*total_iter)
        SHOW_META_LOSS = True
    else:
        do_meta_after_iter = -1
        SHOW_META_LOSS = False

    # ITERATION LOOP
    # for iter_ind in range(start_iter, total_iter+1):
    for iter_ind, inputs in zip(range(start_iter, total_iter+1), data_loader):

        if config.DEV.SWITCH and not config.DEV.BASELINE:
            if iter_ind > do_meta_after_iter:
                do_meta = True
                if SHOW_META_LOSS:
                    print_log('\n** Do meta loss from ep {}, iter {} **\n'.format(curr_ep, iter_ind), config.MISC.LOG_FILE)
                    SHOW_META_LOSS = False
            else:
                do_meta = False

        curr_iter_time_start = time.time()
        lr = adjust_lr(optimizer, curr_ep, iter_ind, config.TRAIN)   # return lr to show in console

        # takes super long time!!!
        # (when bs is large, like 32, use iterator costs 27s while use zip takes 0.0x seconds)
        # inputs = next(data_iterator)
        images = Variable(inputs[0].cuda())
        image_metas = Variable(inputs[-1].cuda())
        # print('fetch data time: {:.4f}'.format(time.time() - curr_iter_time_start))

        # pad with zeros
        gt_class_ids, gt_boxes, gt_masks, _ = model.adjust_input_gt(inputs[1], inputs[2], inputs[3])

        if config.CTRL.PROFILE_ANALYSIS:
            print('\ncurr_iter: ', iter_ind)
            print('fetch data time: {:.4f}'.format(time.time() - curr_iter_time_start))
            t = time.time()
        try:
            # FORWARD PASS
            # the loss shape: gpu_num x 5; meta_loss *NOT* included
            merged_loss, \
            big_feat, big_cnt, small_feat, small_cnt, big_loss, \
            small_output_all, small_gt_all, fpn_ot_loss = \
                input_model([images, gt_class_ids, gt_boxes, gt_masks, image_metas], 'train')
        except Exception:
            info_pass = {
                'type': 'Runtime Error',
                'curr_ep': curr_ep,
                'iter_ind': iter_ind,
            }
            if config.MISC.USE_VISDOM:
                vis.show_dynamic_info(**info_pass)
            raise RuntimeError('whoops, some error pops up...')

        detailed_loss = torch.mean(merged_loss, dim=0)

        # meta-loss
        if config.DEV.SWITCH and not config.DEV.BASELINE:
            if config.DEV.DIS_REG_LOSS:
                detailed_loss.data[3] = 0  # roi_bbox
                detailed_loss.data[1] = 0  # rpn_bbox
                detailed_loss.data[4] = 0  # mask

            # big_feat/small_feat: gpu_num x scale_num x 1024 x 81; also update the buffer
            if small_feat.sum().data[0] != 0:
                meta_loss = model.meta_loss([big_feat, big_cnt, small_feat, small_cnt,
                                             small_output_all, small_gt_all])
            else:
                meta_loss = Variable(torch.zeros(1).cuda())

            _meta_loss_value = meta_loss.data.cpu()[0]
            if _meta_loss_value < 0:
                print_log('\n** meta_loss: {:.4f}, at iter {:d} epoch {:d}; set to 0 in this case **\n'.format(
                    _meta_loss_value, iter_ind, curr_ep), config.MISC.LOG_FILE)
                meta_loss = Variable(torch.zeros(1).cuda())

            if do_meta:
                meta_loss *= config.DEV.LOSS_FAC
            else:
                # for the very first few iter, we don't compute meta-loss
                # but rather accumulate the buffer pool
                meta_loss = Variable(torch.zeros(1).cuda())
        else:
            meta_loss = 0

        # big-loss
        if config.DEV.SWITCH and config.DEV.BIG_SUPERVISE:
            # big loss: gpu_num x scale_num x 1
            big_loss = torch.mean(big_loss)
            big_loss *= config.DEV.BIG_LOSS_FAC
        else:
            big_loss = 0

        # final loss
        fpn_ot_loss_avg = config.TRAIN.FPN_OT_LOSS_FAC * torch.mean(fpn_ot_loss)
        loss = torch.sum(detailed_loss) + meta_loss + big_loss + fpn_ot_loss_avg
        if config.CTRL.PROFILE_ANALYSIS:
            print('forward time: {:.4f}'.format(time.time() - t))
            t = time.time()

        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm(input_model.parameters(), config.TRAIN.MAX_GRAD_NORM)
        optimizer.step()

        if config.CTRL.PROFILE_ANALYSIS:
            print('backward time: {:.4f}'.format(time.time() - t))
            t = time.time()

        # Progress
        if iter_ind % config.CTRL.SHOW_INTERVAL == 0 \
                or iter_ind == args['start_iter'] or iter_ind == total_iter:
            info_pass = {
                'type': 'Regular',
                'curr_iter_time_start': curr_iter_time_start,
                'curr_ep': curr_ep,
                'iter_ind': iter_ind,
                'total_iter': total_iter,
                'meta_loss': meta_loss,
                'big_loss': big_loss,
                'loss': loss,
                'fpn_ot_loss': fpn_ot_loss_avg,
                'lr': lr,
                'detailed_loss': detailed_loss,
                'stage_name': args['stage_name'],
                'epoch_str': args['epoch_str'],
            }
            show_loss_terminal(config, **info_pass)
            if config.MISC.USE_VISDOM:
                loss_data = vis.plot_loss(**info_pass)
                vis.show_dynamic_info(**info_pass)

        # save model
        if iter_ind % save_iter_base == 0:
            if not config.MISC.USE_VISDOM:
                loss_data = []
            info_pass = {
                'epoch':        curr_ep,        # or model.epoch
                'iter':         iter_ind,       # or model.iter
                'loss_data':    loss_data
            }
            save_model(model, **info_pass)

    return loss_data


def test_model(input_model, valset, coco_api, limit=-1, image_ids=None, **args):
    """
        Test the trained model
        Args:
            input_model:    nn.DataParallel
            valset:         validation dataset
            coco_api:       api
            limit:          the number of images to use for evaluation
            image_ids:      a certain image
    """
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    vis = args['vis']
    # set up save and log folder for both train and inference
    if args['during_train']:
        model_file_name = 'mask_rcnn_ep_{:04d}_iter_{:06d}.pth'.format(args['epoch'], args['iter'])
        mode = 'inference'
        _val_folder = model.config.MISC.RESULT_FOLDER.replace('train', 'inference')
        _model_name = model_file_name.replace('.pth', '')
        _model_suffix = _model_name.replace('mask_rcnn_', '')  # say, ep_0053_iter_1234
        log_file = os.path.join(_val_folder, 'inference_from_{:s}.txt'.format(_model_name))
        if not os.path.exists(_val_folder):
            os.makedirs(_val_folder)
        det_res_file = os.path.join(_val_folder, 'det_result_{:s}.pth'.format(_model_suffix))
        train_log_file = model.config.MISC.LOG_FILE
        save_im_folder = os.path.join(_val_folder, _model_suffix)
        if model.config.TEST.SAVE_IM:
            if not os.path.exists(save_im_folder):
                os.makedirs(save_im_folder)
        now = datetime.datetime.now()
        print_log('\n[Inference during train] Start timestamp: {:%Y%m%dT%H%M}'.format(now), file=log_file, init=True)
    else:
        # validation/visualization case
        model_file_name = os.path.basename(model.config.MODEL.INIT_MODEL)
        mode = model.config.CTRL.PHASE   # could be inference or visualize

        log_file = model.config.MISC.LOG_FILE
        det_res_file = model.config.MISC.DET_RESULT_FILE

        vis_res_folder = model.config.MISC.VIS_RESULT_FOLDER
        _name = 'features_afew.pth' if model.config.TSNE.A_FEW else 'features.pth'
        # results/meta_105_quick_1_roipool/visualize/vis_result_ep_0013_iter_000619/features.pth
        vis_file_name = os.path.join(vis_res_folder, _name)
        vis_res_figure = model.config.TSNE.VIS_RES_FIGURE

        train_log_file = None
        save_im_folder = model.config.MISC.SAVE_IMAGE_DIR if model.config.TEST.SAVE_IM else None

    dataset = valset.dataset
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids
    # Limit to a subset
    if limit > 0:
        image_ids = image_ids[:limit]

    num_test_im = len(image_ids)
    test_bs = model.config.TEST.BATCH_SIZE
    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[ind]["id"] for ind in image_ids]
    t_prediction = 0
    t_start = time.time()

    skip = False
    if det_res_file is not None and os.path.exists(det_res_file):
        print_log('results file: {} exists, skip inference and directly evaluate ...'.format(det_res_file),
                  log_file, additional_file=train_log_file)
        results = torch.load(det_res_file)['det_result']
        skip = True

    if vis_file_name is not None and os.path.exists(vis_file_name):
        print_log('results file: {} exists, skip visualize and directly TSNE ...'.format(vis_file_name),
                  log_file, additional_file=train_log_file)
        results = torch.load(vis_file_name)['feat_result']
        skip = True

    # inference: extract features, do detections
    if not skip:
        print_log("Running COCO evaluation on {} images.".format(num_test_im), log_file, additional_file=train_log_file)
        assert (num_test_im % test_bs) % model.config.MISC.GPU_COUNT == 0, \
            '[INFERENCE/VISUALIZE] last mini-batch in an epoch is not divisible by gpu number.'

        results, cnt = [], 0
        total_iter = math.ceil(num_test_im / test_bs)
        if mode == 'visualize' and model.config.TSNE.A_FEW:
            total_iter = 20
            num_test_im = total_iter * test_bs

        show_test_progress_base = math.floor(total_iter / (model.config.CTRL.SHOW_INTERVAL/2))
        # note that GPU efficiency is low when SAVE_IM=True
        for iter_ind in range(total_iter):

            curr_start_id = iter_ind*test_bs
            curr_end_id = min(curr_start_id + test_bs, num_test_im)
            curr_image_ids = image_ids[curr_start_id:curr_end_id]

            # Run detection
            t_pred_start = time.time()
            # Mold inputs to format expected by the neural network
            molded_images, image_metas, windows, images = _mold_inputs(model, curr_image_ids, dataset)

            # FORWARD PASS
            if mode == 'inference':
                # detections: 8,100,6; mrcnn_mask: 8,100,81,28,28
                detections, mrcnn_mask = input_model([molded_images, image_metas], mode=mode)
            elif mode == 'visualize':
                # out_feat: 8,100,1024
                detections, out_feat = input_model([molded_images, image_metas], mode=mode)
                out_feat = out_feat.data.cpu().numpy()

            # Convert to numpy
            detections = detections.data.cpu().numpy()
            if mode == 'inference':
                mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).contiguous().data.cpu().numpy()

            # LOOP for each image within this batch
            for i, image in enumerate(images):

                curr_coco_id = coco_image_ids[curr_image_ids[i]]
                input_value = mrcnn_mask[i] if mode == 'inference' else out_feat[i]

                final_rois, final_class_ids, final_scores, output_value = _unmold_detections(
                    detections[i], input_value, image.shape, windows[i], mode == 'inference')

                if final_rois is None:
                    continue
                for det_id in range(final_rois.shape[0]):
                    # EACH INSTANCE
                    bbox = np.around(final_rois[det_id], 1)
                    if mode == 'inference':
                        final_masks = output_value
                        curr_result = {
                            "image_id":     curr_coco_id,
                            "category_id":  dataset.get_source_class_id(final_class_ids[det_id], "coco"),
                            "bbox":         [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                            "score":        final_scores[det_id],
                            "segmentation": maskUtils.encode(np.asfortranarray(final_masks[:, :, det_id]))
                        }
                    elif mode == 'visualize':
                        final_feat = output_value[det_id]
                        curr_result = {
                            "image_id": curr_coco_id,
                            "category_id": dataset.get_source_class_id(final_class_ids[det_id], "coco"),
                            "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                            "score": final_scores[det_id],
                            "feature": final_feat
                        }
                    results.append(curr_result)
                # visualize result if necessary
                if model.config.TEST.SAVE_IM:
                    plt.close()
                    display_instances(
                        image, final_rois, final_masks, final_class_ids, CLASS_NAMES, final_scores)

                    im_file = os.path.join(save_im_folder, 'coco_im_id_{:d}.png'.format(curr_coco_id))
                    plt.savefig(im_file, bbox_inches='tight')

            t_prediction += (time.time() - t_pred_start)
            cnt += len(curr_image_ids)

            # show progress
            if iter_ind % show_test_progress_base == 0 or cnt == num_test_im:
                print_log('[{:s}][{:s}] {:s} progress \t{:4d} images /{:4d} total ...'.
                          format(model.config.CTRL.CONFIG_NAME, model_file_name, mode, cnt, num_test_im,
                          log_file, additional_file=train_log_file))
        # DONE with the WHOLE EVAL IMAGES

        print_log("Prediction time (inference or visualize): {:.4f}. Average {:.4f} sec/image".format(
            t_prediction, t_prediction / num_test_im), log_file, additional_file=train_log_file)

        if mode == 'inference':
            print_log('Saving results to {:s}'.format(det_res_file), log_file, additional_file=train_log_file)
            torch.save({'det_result': results}, det_res_file)
        elif mode == 'visualize':
            print_log('Saving results to {:s}'.format(vis_file_name), log_file, additional_file=train_log_file)
            torch.save({'feat_result': results}, vis_file_name)

    # evaluate on COCO
    if not model.config.TSNE.SKIP_INFERENCE:
        # Evaluate
        print('\nBegin to evaluate ...')
        # Load results. This modifies results with additional attributes.
        coco_results = coco_api.loadRes(results)
        eval_type = "bbox"
        coco_eval = COCOeval(coco_api, coco_results, eval_type)
        coco_eval.params.imgIds = coco_image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize(log_file)
        mAP = coco_eval.stats[0]

        print_log('Total time: {:.4f}'.format(time.time() - t_start), log_file, additional_file=train_log_file)
        print_log('Config_name [{:s}], model file [{:s}], mAP is {:.4f}\n\n'.
                  format(model.config.CTRL.CONFIG_NAME, model_file_name, mAP),
                  log_file, additional_file=train_log_file)
        print_log('Done!', log_file, additional_file=train_log_file)
        if model.config.MISC.USE_VISDOM:
            vis.show_mAP(model_file=model_file_name, mAP=mAP)


def _mold_inputs(model, image_ids, dataset):
    """
        FOR EVALUATION ONLY.
        Takes a list of images and modifies them to the format expected as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have different sizes.

        Returns 3 Numpy matrices:
            molded_images: [N, h, w, 3]. Images resized and normalized.
            image_metas: [N, length of meta datasets]. Details about each image.
            windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    images = []

    for curr_id in image_ids:
        image = dataset.load_image(curr_id)
        # Resize image to fit the model expected size
        molded_image, window, scale, padding = resize_image(
            image, min_dim=model.config.DATA.IMAGE_MIN_DIM,
            max_dim=model.config.DATA.IMAGE_MAX_DIM, padding=model.config.DATA.IMAGE_PADDING)
        molded_image = molded_image.astype(np.float32) - model.config.DATA.MEAN_PIXEL

        # Build image_meta
        image_meta = compose_image_meta(0, image.shape, window,
                                        np.zeros([model.config.DATASET.NUM_CLASSES], dtype=np.int32), 0)
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
        images.append(image)

    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)

    # Convert images to torch tensor
    molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()
    molded_images = Variable(molded_images.cuda(), volatile=True)
    image_metas = Variable(torch.from_numpy(image_metas).cuda(), volatile=True)

    return molded_images, image_metas, windows, images


def _unmold_detections(detections, input_value, image_shape, window, inference=True):
    """
        FOR EVALUATION ONLY.
        Re-formats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

            detections:     [100, (y1, x1, y2, x2, class_id, score)]
            input_value:
                            mrcnn_mask:     [100, height, width, num_classes]
                            OR
                            feature:        [100, 1025]

            image_shape:    [height, width, depth] Original size of the image before resizing
            window:         [y1, x1, y2, x2] Box in the image where the real image is excluding the padding.

        Returns:
            boxes:          [N (<=100; actual no. of detections), (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids:      [N] Integer class IDs for each bounding box
            scores:         [N] Float probability scores of the class_id
            output_value:
                            masks:          [height, width, num_instances] Instance masks
                            OR
                            final_feature
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    if inference:
        masks = input_value[np.arange(N), :, :, class_ids]
    else:
        feature = input_value[:N]

    # Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

    # Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

    # **FILTER OUT** detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        if inference:
            masks = np.delete(masks, exclude_ix, axis=0)
        else:
            feature = np.delete(feature, exclude_ix, axis=0)

    if inference:
        N = class_ids.shape[0]
        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty((0,) + masks.shape[1:3])
        output_value = full_masks
    else:
        area = (boxes[:, 0] - boxes[:, 2]) * (boxes[:, 1] - boxes[:, 3]) / (image_shape[0]*image_shape[1])
        output_value = np.concatenate((feature, np.expand_dims(area, axis=1)), axis=1)

    return boxes, class_ids, scores, output_value

