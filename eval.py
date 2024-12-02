import os
import cv2
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, parse_devices
from utils.visualize import print_iou
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.model import MRFS
from dataloader.dataloader import ValPre

logger = get_logger()

def get_class_colors():
    pattale = [
        [0, 0, 0],  # unlabelled
        [128, 0, 64],  # car
        [0, 64, 64],  # person
        [192, 128, 0],  # bike
        [192, 0, 0],  # curve
        [0, 128, 128],  # car_stop
        [128, 64, 64],  # guardrail
        [128, 128, 192],  # color_cone
        [0, 64, 192],  # bump
    ]
    return pattale

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        pred = self.sliding_eval_rgbX(img, modal_x, self.eval_crop_size, self.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.class_num, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)

            fn = name + '.png'

            # save colored result
            class_colors = self.config.pattale # get_class_colors()
            temp = np.zeros((pred.shape[0], pred.shape[1], 3))
            ground_truth = np.zeros((pred.shape[0], pred.shape[1], 3))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # 白色文本
            thickness = 2
            line_type = cv2.LINE_AA

            for i in range(self.class_num):
                temp[pred == i] = class_colors[i]
                ground_truth[label == i] = class_colors[i]

            cv2.putText(temp, 'Prediction', (10, 30), font, font_scale, font_color, thickness, line_type)
            cv2.putText(ground_truth, 'Ground Truth', (10, 30), font, font_scale, font_color, thickness, line_type)

            result = np.hstack((temp, ground_truth))
            cv2.imwrite(os.path.join(self.save_path, fn), result)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.class_num, self.class_num))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc, class_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

if __name__ == "__main__":
    dataset_name = "MFNet"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='MRFS', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=f"./results/{dataset_name}")
    # Dataset Config
    parser.add_argument('--dataset_path', default="../MMFS/datasets/MFNet", type=str, help='absolute path of the dataset root')
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
    log_dir = f"./experiment/MFNet/exp1"#"./checkpoints/log_{dataset_name}_{backbone}"
    parser.add_argument('--log_dir', default=log_dir, type=str, help=' ')
    parser.add_argument('--log_dir_link', default=log_dir, type=str, help='')
    parser.add_argument('--checkpoint_dir', default=os.path.join(log_dir, "checkpoint"), type=str, help='')
    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    parser.add_argument('--log_file', default=os.path.join(log_dir, f"val_{exp_time}.log"), type=str, help='')
    parser.add_argument('--link_log_file', default=os.path.join(log_dir, "val_last.log"), type=str, help='')


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
        segmentor = SegEvaluator(config, dataset, network, all_dev, config.verbose, args.save_path)
        segmentor.run(args.checkpoint_dir, args.epochs, args.log_file, args.link_log_file)