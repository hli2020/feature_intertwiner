import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.nn as nn
import datetime
import time

import os
import math
import yaml
import copy
from tools.collections import AttrDict
from past.builtins import basestring
import numpy as np
from ast import literal_eval
import matplotlib.artist as artist


def cus_set_alpha(e, alpha):
    artist.Artist.set_alpha(e, alpha)
    e._set_facecolor(e._original_facecolor)


def mkdir_if_missing(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def unique1d(variable):
    variable = variable.squeeze()
    assert variable.dim() == 1
    if variable.size(0) == 1:
        return variable
    variable = variable.sort()[0]
    unique_bool = variable[1:] != variable[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if variable.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return variable[unique_bool]


def intersect1d(variable1, variable2):
    aux = torch.cat((variable1, variable2), dim=0)
    aux = aux.squeeze().sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1])]


def log2(x):
    """Implementation of Log2. Pytorch doesn't have a native implementation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass


def print_log(msg, file=None, init=False, additional_file=None, quiet_termi=False):

    if not quiet_termi:
        print(msg)
    if file is None:
        pass
    else:
        if init:
            remove(file)
        with open(file, 'a') as log_file:
            log_file.write('%s\n' % msg)

        if additional_file is not None:
            # TODO (low): a little buggy here: no removal of previous additional_file
            with open(additional_file, 'a') as addition_log:
                addition_log.write('%s\n' % msg)


def compute_left_time(iter_avg, curr_ep, total_ep, curr_iter, total_iter):

    total_time = ((total_iter - curr_iter) + (total_ep - curr_ep)*total_iter) * iter_avg
    days = math.floor(total_time / (3600*24))
    hrs = (total_time - days*3600*24) / 3600
    return days, hrs


def _cls2dict(config):
    output = AttrDict()
    for a in dir(config):
        value = getattr(config, a)
        if not a.startswith("__") and not callable(value):
            assert isinstance(value, AttrDict)
            output[a] = value
    return output


def _dict2cls(_config, config):
    for a in dir(config):
        if not a.startswith("__") and not callable(getattr(config, a)):
            setattr(config, a, _config[a])


def merge_cfg_from_file(cfg_filename, config):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _config = _cls2dict(config)
    _merge_a_into_b(yaml_cfg, _config)
    _dict2cls(_config, config)


def merge_cfg_from_list(cfg_list, config):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    _config = _cls2dict(config)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = _config
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value
    _dict2cls(_config, config)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


# =========== MODEL UTILITIES ===========
def _find_last(config):

    dir_name = os.path.join('results', config.CTRL.CONFIG_NAME.lower(), 'train')
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint


def update_config_and_load_model(config, model, train_generator=None):
    """model should not be DataParallel"""
    choice = config.MODEL.INIT_FILE_CHOICE
    phase = config.CTRL.PHASE

    # 1. determine model_path to load model
    use_pretrain = False
    if phase == 'train':
        if os.path.exists(choice):
            print('[{:s}]loading designated weights\t{:s}\n'.format(phase.upper(), choice))
            model_path = choice
            del config.MODEL['PRETRAIN_COCO_MODEL']
            del config.MODEL['PRETRAIN_IMAGENET_MODEL']
        else:
            model_path = _find_last(config)[1]
            if model_path is not None:
                if choice.lower() in ['coco_pretrain', 'imagenet_pretrain']:
                    print('WARNING: find existing model... ignore pretrain model')

                del config.MODEL['PRETRAIN_COCO_MODEL']
                del config.MODEL['PRETRAIN_IMAGENET_MODEL']
            else:
                if choice.lower() == "imagenet_pretrain":
                    model_path = config.MODEL.PRETRAIN_IMAGENET_MODEL
                    suffix = 'imagenet'
                    del config.MODEL['PRETRAIN_COCO_MODEL']
                elif choice.lower() == "coco_pretrain":
                    model_path = config.MODEL.PRETRAIN_COCO_MODEL
                    suffix = 'coco'
                    del config.MODEL['PRETRAIN_IMAGENET_MODEL']
                elif choice.lower() == 'last':
                    model_path = config.MODEL.PRETRAIN_COCO_MODEL
                    suffix = 'coco'
                    del config.MODEL['PRETRAIN_IMAGENET_MODEL']
                    print('init file choice is [LAST]; however no file found; '
                          'use pretrain model to init')
                print('use {:s} pretrain model...'.format(suffix))
                use_pretrain = True

        print('loading weights \t{:s}\n'.format(model_path))
    elif phase == 'inference' or 'visualize':
        del config.MODEL['PRETRAIN_COCO_MODEL']
        del config.MODEL['PRETRAIN_IMAGENET_MODEL']

        if choice.lower() in ['coco_pretrain', 'imagenet_pretrain', 'last']:
            model_path = _find_last(config)[1]
            print('use last trained model for inference')
        elif os.path.exists(choice):
            model_path = choice
            print('use designated model for inference')
        print('[{:s}] loading model weights\t{:s} for {}\n'.format(phase.upper(), model_path, phase.upper()))

    if not os.path.exists(model_path):
        raise Exception('For now we do not allow training from scratch!!!')
    # set MODEL.INIT_MODEL
    config.MODEL.INIT_MODEL = model_path

    # 2. LOAD MODEL (resumed or pretrain, all phases)
    checkpoints = torch.load(model_path)
    try:
        model.load_state_dict(checkpoints['state_dict'], strict=False)
    except KeyError:
        model.load_state_dict(checkpoints, strict=False)  # legacy reason or pretrain model

    # 3. determine start_iter and epoch for resume or display (for all phases)
    # update network.start_epoch, network.start_iter
    try:
        # indicate this is a resumed model
        model.start_epoch = checkpoints['epoch']
        model.start_iter = checkpoints['iter']
        num_train_im = train_generator.dataset.dataset.num_images
        iter_per_epoch = math.floor(num_train_im/config.TRAIN.BATCH_SIZE)
        if model.start_iter % iter_per_epoch == 0:
            model.start_iter = 1
            model.start_epoch += 1
        else:
            model.start_iter += 1
    except KeyError:
        # indicate this is a pretrain model
        model.start_epoch, model.start_iter = 1, 1
    if config.TRAIN.FORCE_START_EPOCH:
        model.start_epoch, model.start_iter = config.TRAIN.FORCE_START_EPOCH, 1
    # init counters
    model.epoch = model.start_epoch
    model.iter = model.start_iter

    if phase == 'train':
        # 3.1 load previous loss data in Visdom (for train only)
        try:
            loss_data = checkpoints['loss_data']  # could be empty list [] or dict()
            # resumed model
            if config.MISC.USE_VISDOM:
                assert isinstance(loss_data, dict)
                model.start_loss_data = loss_data
        except KeyError:
            pass

    now = datetime.datetime.now()
    # 4. set MISC.LOG_FILE;
    # for inference, set also MISC.DET_RESULT_FILE, MISC.SAVE_IMAGE_DIR
    if phase == 'train':
        config.MISC.LOG_FILE = os.path.join(
            config.MISC.RESULT_FOLDER, 'train_log_start_ep_{:04d}_iter_{:06d}.txt'.
                format(model.start_epoch, model.start_iter))
        print_log('\nStart timestamp: {:%Y%m%dT%H%M}'.format(now), file=config.MISC.LOG_FILE, init=True)
        if config.CTRL.DEBUG:
            # set SAVE_IM=True when debug
            # update: we no longer save_im when TRAIN.DO_VALIDATION=True
            config.TEST.SAVE_IM = True

        # 4.1 set up buffer for meta-loss
        if config.DEV.SWITCH and not config.DEV.BASELINE:
            try:
                # indicate this is a resumed model
                model.buffer = torch.from_numpy(checkpoints['buffer']).cuda()
                model.buffer_cnt = torch.from_numpy(checkpoints['buffer_cnt']).cuda()
                buffer_size = model.buffer.size(0)
                if buffer_size != config.DEV.BUFFER_SIZE:
                    print_log('[WARNING] loaded buffer size: {}, config size: {}\n'
                              'check your config; for now initialize buffer as instructed in config when resume!!!'.
                              format(buffer_size, config.DEV.BUFFER_SIZE), config.MISC.LOG_FILE)
                    model.initialize_buffer(config.MISC.LOG_FILE)
                else:
                    print_log('load existent buffer from previous model ...', config.MISC.LOG_FILE)

            except KeyError:
                model.initialize_buffer(config.MISC.LOG_FILE)
                # indicate this is a pretrain model; init buffer as instructed in config

    elif phase == 'inference' or phase == 'visualize':

        tiny_diff = 'inference' if phase == 'inference' else 'visualize'
        model_name = os.path.basename(model_path).replace('.pth', '')   # mask_rcnn_ep_0053_iter_001234
        model_suffix = os.path.basename(model_path).replace('mask_rcnn_', '')

        if phase == 'inference':
            config.MISC.LOG_FILE = os.path.join(config.MISC.RESULT_FOLDER,
                                                '{:s}_from_{:s}.txt'.format(tiny_diff, model_name))
            config.MISC.DET_RESULT_FILE = os.path.join(config.MISC.RESULT_FOLDER,
                                                       'det_result_{:s}'.format(model_suffix))
        elif phase == 'visualize':
            # NOTE: it is called *folder*; not file!
            # results/meta_105_quick_1_roipool/visualize/vis_result_ep_0013_iter_000619/
                # features.pth
                # set_1_xx/
                #       scatter_xx.png
                #       log.txt file HERE
            config.MISC.VIS_RESULT_FOLDER = os.path.join(config.MISC.RESULT_FOLDER,
                                                         'vis_result_{}'.format(model_suffix)).replace('.pth', '')
            # figure folder to save
            _suffix = '' if config.TSNE.FIG_FOLDER_SUX == '' else '_{}'.format(config.TSNE.FIG_FOLDER_SUX)
            config.TSNE.VIS_RES_FIGURE = os.path.join(config.MISC.VIS_RESULT_FOLDER, '{}_bs_{}{}'.format(
                config.TSNE.SAMPLE_CHOICE, config.TSNE.BATCH_SZ, _suffix))
            mkdir_if_missing(config.TSNE.VIS_RES_FIGURE)

            config.MISC.LOG_FILE = os.path.join(config.TSNE.VIS_RES_FIGURE, 'tsne_train_log.txt')

        print_log('\nStart timestamp: {:%Y%m%dT%H%M}'.format(now), file=config.MISC.LOG_FILE, init=True)

        if config.TEST.SAVE_IM:
            config.MISC.SAVE_IMAGE_DIR = os.path.join(config.MISC.RESULT_FOLDER, model_suffix.replace('.pth', ''))
            if not os.path.exists(config.MISC.SAVE_IMAGE_DIR):
                os.makedirs(config.MISC.SAVE_IMAGE_DIR)

    # 5. show pretrain details
    if use_pretrain:
        print_log('\tuse pretrain_model; pretrain weights detail in log file; NOT shown in terminal',
                  config.MISC.LOG_FILE)
        for key, value in sorted(checkpoints.items()):
            print_log('\t\t{}, size: {}'.format(key, value.size()), config.MISC.LOG_FILE, quiet_termi=True)

        missing = set(model.state_dict().keys()) - set(checkpoints.keys())
        if len(missing) > 0:
            print_log('\tsome layers are trained from scratch; NO weights available in pretrain model. They are:',
                      config.MISC.LOG_FILE)
            for key in sorted(missing):
                print_log('\t\t{:s}, {}'.format(key, model.state_dict()[key].size()), config.MISC.LOG_FILE)

        # 5.1 use some layer weights from pretrain to the new model even their names are different
        if config.DEV.BIG_SUPERVISE and config.DEV.BIG_FC_INIT == 'coco_pretrain':
            _load_state_dict_anyway(model, checkpoints, config.DEV.BIG_FC_INIT_LIST, config.MISC.LOG_FILE)

    if config.MISC.USE_VISDOM:
        print_log('see configurations in Visdom or log file; NOT shown in terminal.', config.MISC.LOG_FILE)
        config.display(config.MISC.LOG_FILE, quiet=True)
    else:
        config.display(config.MISC.LOG_FILE)

    model.config = config
    return config, model


def _load_state_dict_anyway(model, state_dict, map_list, log_file=None):
    # referenced from load_state_dict() in module.py
    own_state = model.state_dict()
    for target_name, pretrain_name in map_list.items():
        pretrain_param = state_dict[pretrain_name]
        if isinstance(pretrain_param, Parameter):
            # backwards compatibility for serialized parameters
            pretrain_param = pretrain_param.data
        try:
            own_state[target_name].copy_(pretrain_param)
            print_log('\t[DELIBERATE COPY] weights in pretrain layer [{:s}] to layer [{:s}]'
                      .format(pretrain_name, target_name), log_file)
        except Exception:
            raise RuntimeError('While copying the parameter named {}, '
                               'whose dimensions in the model are {} and '
                               'whose dimensions in the checkpoint are {}.'
                               .format(pretrain_name, own_state[target_name].size(), pretrain_param.size()))


def set_optimizer(net, opt):

    if opt.OPTIM_METHOD == 'sgd':

        if opt.BN_LEARN:
            parameter_list = [param for name, param in net.named_parameters() if param.requires_grad]
            optimizer = optim.SGD(parameter_list, lr=opt.INIT_LR,
                                  momentum=opt.MOMENTUM, weight_decay=opt.WEIGHT_DECAY)
        else:
            # Optimizer object, add L2 Regularization
            # Skip regularization of gamma and beta weights in batch normalization layers.
            trainables_wo_bn = [param for name, param in net.named_parameters()
                                if param.requires_grad and 'bn' not in name]
            trainables_only_bn = [param for name, param in net.named_parameters()
                                  if param.requires_grad and 'bn' in name]
            optimizer = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': opt.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=opt.INIT_LR, momentum=opt.MOMENTUM)

    elif opt.OPTIM_METHOD == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.INIT_LR,
                               weight_decay=opt.WEIGHT_DECAY, betas=(0.9, 0.999))
    elif opt.OPTIM_METHOD == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr,
                                  weight_decay=opt.WEIGHT_DECAY, momentum=opt.MOMENTUM,
                                  alpha=0.9, centered=True)
    return optimizer


def adjust_lr(optimizer, curr_ep, curr_iter, config):

    if config.LR_WARM_UP and curr_ep == 1 and curr_iter <= config.LR_WP_ITER:
        a = config.INIT_LR * (1 - config.LR_WP_FACTOR) / (config.LR_WP_ITER - 1)
        b = config.INIT_LR * config.LR_WP_FACTOR - a
        lr = a * curr_iter + b
    else:
        def _tiny_transfer(schedule):
            out = np.zeros(len(schedule))
            for i in range(len(schedule)):
                out[i] = sum(schedule[:i+1])
            return out
        schedule_list = _tiny_transfer(config.SCHEDULE)
        decay = config.GAMMA ** (sum(curr_ep > schedule_list))
        lr = config.INIT_LR * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def show_loss_terminal(config, **args):

    curr_iter_time_start = args['curr_iter_time_start']
    curr_ep, iter_ind, total_iter = args['curr_ep'], args['iter_ind'], args['total_iter']
    loss = args['loss']
    lr = args['lr']
    detailed_loss = args['detailed_loss']
    stage_name, epoch_str = args['stage_name'], args['epoch_str']

    iter_time = time.time() - curr_iter_time_start
    days, hrs = compute_left_time(
        iter_time, curr_ep, sum(config.TRAIN.SCHEDULE), iter_ind, total_iter)

    # additional loss fill up here
    meta_loss = args['meta_loss']
    big_loss = args['big_loss']
    fpn_ot_loss = args['fpn_ot_loss']
    suffix = ' - meta_loss: {:.3f}' if config.DEV.SWITCH and not config.DEV.BASELINE else '{:s}'
    suffix += ' - big_loss: {:.3f}' if config.DEV.SWITCH and config.DEV.BIG_SUPERVISE else '{:s}'
    suffix += ' - fpn_ot: {:.3f}' if config.TRAIN.FPN_OT_LOSS else '{:s}'
    last_out_1 = meta_loss.data.cpu()[0] if config.DEV.SWITCH and not config.DEV.BASELINE else ''
    last_out_2 = big_loss.data.cpu()[0] if config.DEV.SWITCH and config.DEV.BIG_SUPERVISE else ''
    last_out_3 = fpn_ot_loss.data.cpu()[0] if config.TRAIN.FPN_OT_LOSS else ''

    progress_str = '[{:s}][{:s}]{:s} {:06d}/{} [est. left: {:d} days, {:.2f} hrs] (iter_t: {:.2f})' \
                   '\tlr: {:.6f} | loss: {:.3f} - rpn_cls: {:.3f} - rpn_bbox: {:.3f} ' \
                   '- mrcnn_cls: {:.3f} - mrcnn_bbox: {:.3f} - mrcnn_mask_loss: {:.3f}' + suffix

    config_name_str = config.CTRL.CONFIG_NAME if not config.CTRL.QUICK_VERIFY else \
        config.CTRL.CONFIG_NAME + ', quick verify mode'

    print_log(progress_str.format(
        config_name_str, stage_name, epoch_str, iter_ind, total_iter,
        days, hrs, iter_time, lr,
        loss.data.cpu()[0],
        detailed_loss[0].data.cpu()[0], detailed_loss[1].data.cpu()[0],
        detailed_loss[2].data.cpu()[0], detailed_loss[3].data.cpu()[0],
        detailed_loss[4].data.cpu()[0],
        last_out_1, last_out_2, last_out_3),
        config.MISC.LOG_FILE)


def save_model(model, **args):
    config = model.config
    curr_ep, iter_ind = args['epoch'], args['iter']
    loss_data = args['loss_data']

    model_file = os.path.join(
        config.MISC.RESULT_FOLDER, 'mask_rcnn_ep_{:04d}_iter_{:06d}.pth'.format(curr_ep, iter_ind))
    print_log('saving model: {:s}\n'.format(model_file), config.MISC.LOG_FILE)
    if config.DEV.SWITCH and not config.DEV.BASELINE:  # has meta-loss
        buffer, buffer_cnt = model.buffer.cpu().numpy(), model.buffer_cnt.cpu().numpy()
    else:
        buffer, buffer_cnt = [], []
    torch.save({
        'state_dict':   model.state_dict(),
        'epoch':        curr_ep,        # or model.epoch
        'iter':         iter_ind,       # or model.iter
        'buffer':       buffer,
        'buffer_cnt':   buffer_cnt,
        'loss_data':    loss_data
    }, model_file)


def check_max_mem(input_model, data_loader, MaskRCNN):

    config = input_model.config
    input_model = set_model(config.MISC.GPU_COUNT, input_model)

    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    print_log('\nchecking possibly MAX mem cost ...', config.MISC.LOG_FILE)
    # set optimizer
    optimizer = set_optimizer(model, config.TRAIN)
    model.buffer = torch.zeros(model.config.DEV.BUFFER_SIZE, 1024, config.DATASET.NUM_CLASSES).cuda()
    model.buffer_cnt = torch.zeros(config.DEV.BUFFER_SIZE, 1, config.DATASET.NUM_CLASSES).cuda()

    for iter_ind, inputs in zip(range(1, 11), data_loader):
        images = Variable(inputs[0].cuda())
        image_metas = Variable(inputs[-1].cuda())
        gt_class_ids, gt_boxes, gt_masks, _ = model.adjust_input_gt(inputs[1], inputs[2], inputs[3])
        merged_loss, big_feat, big_cnt, small_feat, small_cnt, big_loss = \
            input_model([images, gt_class_ids, gt_boxes, gt_masks, image_metas], 'train')
        detailed_loss = torch.mean(merged_loss, dim=0)

        if config.DEV.SWITCH and not config.DEV.BASELINE:
            # big_feat: gpu_num x scale_num x 1024 x 81; also update the buffer
            meta_loss = model.meta_loss([big_feat, big_cnt, small_feat, small_cnt])
            meta_loss *= config.DEV.LOSS_FAC
        else:
            meta_loss = 0

        # big-loss
        if config.DEV.SWITCH and config.DEV.BIG_SUPERVISE:
            # big loss: gpu_num x scale_num x 1
            big_loss = torch.mean(big_loss)
            big_loss *= config.DEV.BIG_LOSS_FAC
        else:
            big_loss = 0

        loss = torch.sum(detailed_loss) + meta_loss + big_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print_log('passed maximum GPU mem test!', config.MISC.LOG_FILE)
    del optimizer, model

    # set optimizer
    model = MaskRCNN(config)
    optimizer = set_optimizer(model, config.TRAIN)
    print_log('set back to original model weights ...\n', config.MISC.LOG_FILE)

    return optimizer, model


def set_model(gpu_cnt, model):
    if gpu_cnt < 1:
        print('cpu mode ...')
    elif gpu_cnt == 1:
        print('single gpu mode ...')
        model = model.cuda()
    else:
        print('multi-gpu mode ...')
        model = nn.DataParallel(model).cuda()
    return model