import argparse
from lib.config import CocoConfig
from lib.workflow import train_model, test_model
from datasets.dataset_coco import get_data
from tools.visualize import Visualizer
from lib.model import MaskRCNN
from tools.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mask R-CNN')

    parser.add_argument('--phase',
                        default='train',
                        # default='inference',
                        help='train or inference')

    parser.add_argument('--config_name',
                        required=False,
                        default='')

    parser.add_argument('--config_file',
                        default='configs/105/meta_105_quick_1.yaml')

    # debug mode: set train_data to val_data for faster data loading.
    # show loss step by step; smaller input image size
    # do validation right after a few steps and visualize predictions
    parser.add_argument('--debug',
                        default=0, type=int)  # no bool type here please

    parser.add_argument('--device_id',
                        default='1', type=str)

    parser.add_argument('opts',
                        help='See lib/config.py for all options',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    print('\nSTART::: phase is [{:s}]'.format(args.phase.upper()))

    # Configuration
    config = CocoConfig(args)
    # Get data
    train_data, val_data, val_api = get_data(config)

    # Create model
    print('building network ...\n')
    model = MaskRCNN(config)

    optimizer = set_optimizer(model, config.TRAIN)

    # Select weights file to load (MUST be put at the end)
    # update start epoch and iter if resume
    config, model = update_config_and_load_model(config, model, train_data)

    # Visualizer
    vis = Visualizer(config, model, val_data)

    print_log('print network structure in log file [NOT shown in terminal] ...', config.MISC.LOG_FILE)
    print_log(model, config.MISC.LOG_FILE, quiet_termi=True)

    model = set_model(config.MISC.GPU_COUNT, model)
    # Train or inference
    if args.phase == 'train':

        # Training - Stage 1
        print("\nTraining network heads")
        train_model(model, train_data, val_data,
                    optimizer=optimizer, layers='heads', coco_api=val_api, vis=vis)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("\nFinetune Resnet stage 4 and up")
        train_model(model, train_data, val_data,
                    optimizer=optimizer, layers='4+', coco_api=val_api, vis=vis)

        # Training - Stage 3
        # Fine tune all layers
        print("\nFine tune all layers")
        train_model(model, train_data, val_data,
                    optimizer=optimizer, layers='all', coco_api=val_api, vis=vis)

    elif args.phase == 'inference' or args.phase == 'visualize':

        test_model(model, val_data, val_api, during_train=False, vis=vis)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.phase))
