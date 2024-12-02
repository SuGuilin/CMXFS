import os
import argparse

import cv2
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, parse_devices
from utils.evaluator_fusion import FuseEvaluator
from engine.evaluator_vision import Evaluator
from kornia.metrics import AverageMeter
from engine.logger import get_logger
from dataloader.RGBXDataset import RGBXDataset
from models.model import MRFS
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()
Fuse_result = [AverageMeter() for _ in range(8)]

class VisionEvaluator(Evaluator):

    def func_per_iteration_vision(self, data, device):
        img = data['data']
        modal_x = data['modal_x']
        name = data['fn']
        Fuse = self.sliding_eval_rgbX_vision(img, modal_x, self.eval_crop_size, self.eval_stride_rate, device)

        Fuse = (Fuse - Fuse.min()) / (Fuse.max() - Fuse.min()) * 255.0
        
        if self.save_path is not None:
            ensure_dir(self.save_path)

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(Fuse.astype(np.uint8), mode='RGB')
            result_img.save(os.path.join(self.save_path, fn))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            modal_x = modal_x[:, :, 0].astype(np.uint8)
            result_img = np.array(result_img.convert('L'))

            Fuse_result[0].update(FuseEvaluator.EN(result_img))
            Fuse_result[1].update(FuseEvaluator.SD(result_img))
            Fuse_result[2].update(FuseEvaluator.SF(result_img))
            Fuse_result[3].update(FuseEvaluator.MI(result_img, modal_x, img))
            Fuse_result[4].update(FuseEvaluator.SCD(result_img, modal_x, img))
            Fuse_result[5].update(FuseEvaluator.VIFF(result_img, modal_x, img))
            Fuse_result[6].update(FuseEvaluator.Qabf(result_img, modal_x, img))
            Fuse_result[7].update(FuseEvaluator.SSIM(result_img, modal_x, img))


if __name__ == "__main__":
    dataset_name = "MFNet"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='MRFS', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=f'./results_Fusion/{dataset_name}')
    # Dataset Config
    parser.add_argument('--dataset_path', default="/home/suguilin/MMFS/datasets/MFNet", type=str, help='absolute path of the dataset root')
    parser.add_argument('--rgb_folder', default="RGB", type=str, help='folder for visible light images')
    parser.add_argument('--rgb_format', default=".png", type=str, help='the load format for visible light images')
    parser.add_argument('--x_folder', default="Modal", type=str, help='folder for thermal imaging images')
    parser.add_argument('--x_format', default=".png", type=str, help='the load format for thermal imaging images')
    parser.add_argument('--x_is_single_channel', default=True, type=bool,
                        help='True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input')
    parser.add_argument('--label_folder', default="Label", type=str, help='folder for segmentation label image')
    parser.add_argument('--label_format', default=".png", type=str, help='the load format for segmentation label image')
    parser.add_argument('--gt_transform', default=False, type=bool, help='')
    parser.add_argument('--num_classes', default=9, type=int, help='')
    parser.add_argument('--class_names',
                        default=['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump'],
                        type=list, help='the class names of all classes')
    # Network Config
    backbone = "vmamba_tiny"
    parser.add_argument('--backbone', default=backbone, type=str, help='the backbone network to load')
    parser.add_argument('--decoder_embed_dim', default=768, type=int, help='')
    # Val Config
    parser.add_argument('--eval_crop_size', default=[480, 640], type=list, help='')
    parser.add_argument('--eval_stride_rate', default=2/3, type=float, help='')
    parser.add_argument('--eval_scale_array', default=[1], type=list, help='')
    parser.add_argument('--is_flip', default=False, type=bool, help='')
    log_dir = f"./experiment/MFNet/exp1"#checkpoints/log_{dataset_name}_{backbone}"
    parser.add_argument('--log_dir', default=log_dir, type=str, help=' ')
    parser.add_argument('--log_dir_link', default=log_dir, type=str, help='')
    parser.add_argument('--checkpoint_dir', default=os.path.join(log_dir, "checkpoint"), type=str, help='')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    import yaml
    from easydict import EasyDict as edict
    def load_config(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return edict(config)
    if dataset_name == 'MFNet':
        config_path = './configs/config_mfnet.yaml'
    elif dataset_name == 'FMB':
        config_path = './configs/config_fmb.yaml'
    else:
        raise ValueError('Not a valid dataset name')

    config = load_config(config_path)
    network = MRFS(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': os.path.join(config.dataset_path, config.rgb_folder),
                    'rgb_format': config.rgb_format,
                    'x_root': os.path.join(config.dataset_path, config.x_folder),
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'gt_root': os.path.join(config.dataset_path, config.label_folder),
                    'gt_format': config.label_format,
                    'transform_gt': config.gt_transform,
                    'class_names': config.class_names,
                    'train_source': os.path.join(config.dataset_path, "train.txt"),
                    'eval_source': os.path.join(config.dataset_path, "test.txt")}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)

    with torch.no_grad():
        Fuser = VisionEvaluator(config, dataset, network, all_dev, config.verbose, args.save_path)
        Fuser.run(args.checkpoint_dir, args.epochs)


    with open(os.path.join(args.log_dir, 'fuse_result.log'), 'w') as f:
        f.write('EN: ' + str(np.round(Fuse_result[0].avg, 3)) + '\n')
        f.write('SD: ' + str(np.round(Fuse_result[1].avg, 3)) + '\n')
        f.write('SF: ' + str(np.round(Fuse_result[2].avg, 3)) + '\n')
        f.write('MI: ' + str(np.round(Fuse_result[3].avg, 3)) + '\n')
        f.write('SCD: ' + str(np.round(Fuse_result[4].avg, 3)) + '\n')
        f.write('VIFF: ' + str(np.round(Fuse_result[5].avg, 3)) + '\n')
        f.write('Qabf: ' + str(np.round(Fuse_result[6].avg, 3)) + '\n')
        f.write('SSIM: ' + str(np.round(Fuse_result[7].avg, 3)) + '\n')


    logger.info(f'writing fusion results done!')
    print("\n" * 2 + "=" * 80)
    print("The fusion test result :")
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(
        'result:\t'
        + '\t'
        + str(np.round(Fuse_result[0].avg, 3))
        + '\t'
        + str(np.round(Fuse_result[1].avg, 3))
        + '\t'
        + str(np.round(Fuse_result[2].avg, 3))
        + '\t'
        + str(np.round(Fuse_result[3].avg, 3))
        + '\t'
        + str(np.round(Fuse_result[4].avg, 3))
        + '\t'
        + str(np.round(Fuse_result[5].avg, 3))
        + '\t'
        + str(np.round(Fuse_result[6].avg, 3))
        + '\t'
        + str(np.round(Fuse_result[7].avg, 3))
    )
    print("=" * 80)