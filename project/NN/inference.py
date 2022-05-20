from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os.path

import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T

from .src.model import SODModel
from .src.dataloader import InfDataloader, SODLoader, pad_resize_image
import torchvision.transforms as transforms
class ImageDataset(Dataset):
    def __init__(self, X, size= 256):
        'Initialization'
        self.X = X
        self.target_size = size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = cv2.cvtColor(self.X, cv2.COLOR_BGR2RGB)
        img_np = pad_resize_image(img, None, self.target_size)
        img_tor = img_np.astype(np.float32)
        img_tor = img_tor / 255.0
        img_tor = np.transpose(img_tor, axes=(2, 0, 1))
        img_tor = torch.from_numpy(img_tor).float()
        img_tor = self.normalize(img_tor)

        return img_np, img_tor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='/data', help='Path to folder containing images', type=str)
    #parser.add_argument('--model_path', default='C:/Users/Michi/Documents/Uni/9.Semester/WS2122_G06/project/NN/model/best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to model', type=str)
    parser.add_argument('--model_path', default='{}/model/best-model_epoch-204_mae-0.0505_loss-0.1370.pth'.format(os.path.dirname(os.path.realpath(__file__))))
    parser.add_argument('--use_gpu', default=False, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)

    return parser.parse_args()


def run_inference(args, image, image_data):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    results = []

    #inf_data = InfDataloader(image, target_size=image.shape[0])
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(image_data, batch_size=1, shuffle=True, num_workers=0)

    #print("Press 'q' to quit.")
    with torch.no_grad():
        for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)

            # Assuming batch_size = 1
            img_np = np.squeeze(img_np.numpy(), axis=0)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

            #print('Image :', batch_idx)
            #cv2.imshow('Input Image', img_np)
            #cv2.imshow('Generated Saliency Mask', pred_masks_raw)
            #cv2.imshow('Rounded-off Saliency Mask', pred_masks_round)
            results.append(pred_masks_round)
            break
            #key = cv2.waitKey(0)
            #if key == ord('q'):
                #break
    return results

def calculate_mae(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    test_data = SODLoader(mode='test', augment_data=False, target_size=args.img_size)
    test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=0)

    # List to save mean absolute error of each image
    mae_list = []
    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(tqdm.tqdm(test_dataloader), start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            pred_masks, _ = model(inp_imgs)

            mae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
            mae_list.extend(mae)

    #print('MAE for the test set is :', np.mean(mae_list))

def get_predictions(img):
    rt_args = parse_arguments()
    image_data = ImageDataset(img)
    calculate_mae(rt_args)
    res = run_inference(rt_args, img, image_data)
    return res[0]
    



