#!/usr/bin/env python

# __author__  = "Chinmoy Samant"
# Work done at Bio Imaging, Signal Processing & Learning Lab under Dr. Jong Chul Ye

import numpy as np
import os, glob, datetime, time, sys, math, argparse
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import scipy.stats as ss
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from loss import *

from dataloader import get_train_valid_loader, get_test_loader
from models import SE_U_Net, SE_U_Net_PP
from utils import dice_coeff

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

ROOT_DIR = os.path.realpath(__file__)
ROOT_DIR = ROOT_DIR.replace("/train.py","")

def train(model, epochs, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, output_dir, model_dir, cuda=True):

    best_score=0
    best_epoch=0
    sigmoid = nn.Sigmoid()

    output_dir_inp_img = output_dir+'inp/'
    output_dir_inp_lbl = output_dir+'lbl/'
    if not os.path.exists(output_dir_inp_img):
        os.makedirs(output_dir_inp_img)
    if not os.path.exists(output_dir_inp_lbl):
        os.makedirs(output_dir_inp_lbl)

    l=0
    for j,(img_vd, lbl_vd) in enumerate(valid_dataloader):
        for k in range(img_vd.shape[0]):
            img_vd_out = Image.fromarray(np.uint8(255*np.transpose(img_vd.data[k], (1,2,0))))
            lbl_vd_out = Image.fromarray(np.uint8(255*lbl_vd.data[k][0]))
            img_vd_out.save(output_dir_inp_img+str(l)+'.jpg', 'JPEG')
            lbl_vd_out.save(output_dir_inp_lbl+str(l)+'.jpg', 'JPEG')
            l=l+1

    for epoch in range(epochs):

        output_dir_ep = output_dir+str(epoch)+'/'
        if not os.path.exists(output_dir_ep):
            os.makedirs(output_dir_ep)

        model.train()
        scheduler.step(epoch)  # step to the learning rate in this epoch
        train_loss = 0
        dice_score = 0

        for i,(img_tr, lbl_tr) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            if cuda:
                img_tr = img_tr.cuda()
                lbl_tr = lbl_tr.cuda()
          
            out_tr = model(img_tr)
            out_pred = sigmoid(out_tr) > 0.5
            loss = criterion(out_tr, lbl_tr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= i+1

        with torch.no_grad():
            l = 0
            for j,(img_vd, lbl_vd) in tqdm(enumerate(valid_dataloader)):
                if(cuda):
                    img_vd = img_vd.cuda()
                    lbl_vd = lbl_vd.cuda()
                
                out_vd = model(img_vd)
                out_pred = sigmoid(out_vd) > 0.5
                dice_score += dice_coeff(out_pred.data, lbl_vd.data)
                for k in range(out_pred.shape[0]):
                    out_pred_img = Image.fromarray(np.uint8(255*out_pred.data)[k][0])
                    out_pred_img.save(output_dir_ep+str(l)+'.jpg', 'JPEG')
                    l=l+1

            dice_score /= j+1

        print('Epoch %4d: Train loss = %2.4f Valid dice_score = %3.6f' % (epoch + 1, train_loss, dice_score))

        if(dice_score > best_score):
            best_score = dice_score
            best_epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

def test(model, test_data, output_dir, cuda=True):

    sigmoid = nn.Sigmoid()
    dice_score = 0

    with torch.no_grad():
        l=0
        for j,(img_vd, lbl_vd) in tqdm(enumerate(test_data)):        
            if(cuda):
                img_vd = img_vd.cuda()
                lbl_vd = lbl_vd.cuda()
            
            out_vd = model(img_vd)
            out_pred = sigmoid(out_vd) > 0.5
            dice_score += dice_coeff(out_pred.data, lbl_vd.data)

            for k in range(out_pred.shape[0]):
                out_pred_img = Image.fromarray(np.uint8(255*out_pred.data)[k][0])
                out_pred_img.save(output_dir+str(l)+'.jpg', 'JPEG')
                l=l+1

        dice_score /= j+1
        print('Test : dice_score = %3.6f' % (dice_score))

def main(args):
    if not os.path.exists(ROOT_DIR + args.valid_out_path):
        os.makedirs(ROOT_DIR + args.valid_out_path)
    if not os.path.exists(ROOT_DIR + args.test_out_path):
        os.makedirs(ROOT_DIR + args.test_out_path)
    if not os.path.exists(ROOT_DIR + args.model_save_path):
        os.makedirs(ROOT_DIR + args.model_save_path)
    
    print('Reading Data')
    loader_train, loader_valid = get_train_valid_loader(data_dir=(ROOT_DIR + args.train_inp_path), batch_size=(args.batch_size_per_gpu*args.num_gpus), random_seed=args.seed,
                                    valid_size=args.valid_size, shuffle=True, num_workers=args.num_workers,
                                    pin_memory=False, train=True, resize=args.resize, resize_h=args.resize_h, resize_w=args.resize_w, crop=args.crop,
                                    crop_h=args.crop_h, crop_w=args.crop_w, hflip=args.h_flip, vflip=args.v_flip)

    laoder_test = get_test_loader(data_dir=ROOT_DIR + args.test_inp_path, batch_size=1*args.num_gpus, shuffle=False, num_workers=1, pin_memory=False, train=False,
                                resize=args.resize, resize_h=args.resize_h, resize_w=args.resize_w)

    # model
    print('Building model')
    if(args.model == 'unetpp'):
        model = SE_U_Net_PP(args.inp_dim, args.out_dim, fc_dim=args.fc_dim, useBN=args.use_BN, useCSE=args.use_CSE, useSSE=args.use_SSE, useCSSE=args.use_CSSE)
    else:
        model = SE_U_Net(args.inp_dim, args.out_dim, fc_dim=args.fc_dim, useBN=args.use_BN, useCSE=args.use_CSE, useSSE=args.use_SSE, useCSSE=args.use_CSSE)

    # criterion = nn.BCELoss()
    # criterion = BoundaryWeightedLoss()
    criterion = nn.BCEWithLogitsLoss()

    if(args.cuda):
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay)  # learning rate scheduler
    
    print('Training model')
    
    best_score = 0
    best_epoch = 0

    train(model=model, epochs=args.num_epoch, cuda=args.cuda, train_dataloader=loader_train, valid_dataloader=loader_valid, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, output_dir=(ROOT_DIR + args.valid_out_path), model_dir=(ROOT_DIR+ args.model_save_path))

    print('Training complete')

    print('Testing model')

    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))

    test(model=model, cuda=args.cuda, test_data=loader_test, output_dir=os.path.join(ROOT_DIR, args.test_out_path))

    print('Testing complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model related arguments
    parser.add_argument('--model', default='unetpp', choices=['unet', 'unetpp'],
                        help="model to be used. Input 'unetpp' for Unet++, 'unet' for standard Unet")
    parser.add_argument('--fc_dim', default=512, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--inp_dim', default=3, type=int,
                        help='number of channels in input image')
    parser.add_argument('--out_dim', default=1, type=int,
                        help='number of channels in output image')
    parser.add_argument('--use_BN', default=True, type=bool,
                        help='whether to use batch normalization or not')
    parser.add_argument('--use_CSE', default=False, type=bool,
                        help='whether to use channel squeeze and excitation block or not')
    parser.add_argument('--use_SSE', default=False, type=bool,
                        help='whether to use spatial squeeze and excitation block or not')
    parser.add_argument('--use_CSSE', default=True, type=bool,
                        help='whether to use channel and spatial squeeze and excitation block or not')

    # Path related arguments
    parser.add_argument('--train_inp_path', type=str,
                        default='/data/train/')
    parser.add_argument('--test_inp_path', type=str,
                        default='/data/test/')
    parser.add_argument('--valid_out_path', type=str,
                        default='/data/valid_output/')
    parser.add_argument('--test_out_path', type=str,
                        default='/data/test_output/')
    parser.add_argument('--model_save_path', type=str,
                        default='/model/')

    # optimization related arguments
    parser.add_argument('--cuda', default='True', type=bool,
                        help='if cuda is supported')
    parser.add_argument('--num_gpus', default=2, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=50, type=int,
                        help='epochs to train for')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='LR')
    parser.add_argument('--steps', default=[30,40], type=float,
                        help='milestones for multi step LR')
    parser.add_argument('--lr_decay', default=0.1, type=float,
                        help='rate of LR decay for multi step LR (gamma)')

    # Data related arguments
    parser.add_argument('--num_class', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--valid_size', default=0.01, type=int,
                        help='training validation data split')
    parser.add_argument('--resize', default=True, type=bool,
                        help='set true to resize images to (resize_h,resize_w)')
    parser.add_argument('--resize_h', default=640, type=int,
                        help='height of resized image')
    parser.add_argument('--resize_w', default=960, type=int,
                        help='width of resized image')
    parser.add_argument('--crop', default=False, type=bool,
                        help='set true to crop images to (crop_h,crop_w)')
    parser.add_argument('--crop_h', default=256, type=int,
                        help='height of cropped image')
    parser.add_argument('--crop_w', default=256, type=int,
                        help='width of cropped image')
    parser.add_argument('--h_flip', default=True, type=bool,
                        help='set true to flip images horizontally with probability 0.5 while training')
    parser.add_argument('--v_flip', default=False, type=bool,
                        help='set true to flip images vertically with probability 0.5 while training')

    # Misc arguments
    parser.add_argument('--seed', default=777, type=int, help='manual seed')

    args = parser.parse_args()

    main(args)