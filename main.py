## ECCV-2018-Audio-Visual Event Localization in Unconstrained Videos
## https://arxiv.org/abs/1803.08842
## supervised audio-visual event localization with feature fusion and audio-guided visual attention

from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report
from dataloader import AVEDataset
import random
from models import Att_Net
random.seed(3344)
import time
import warnings
warnings.filterwarnings('ignore') 
import argparse

parser = argparse.ArgumentParser(description='AVE')
# data specifications
parser.add_argument('--name', type=str, default='AV_att', help='model name')
parser.add_argument('--local', action='store_true', default=False, help='run locally or remotely')
parser.add_argument('--gpu', type=int, default=0, help='gpu selection')
parser.add_argument('--dir_video', type=str, default='visual_feature.h5', help='visual features')
parser.add_argument('--dir_audio', type=str, default='audio_feature.h5', help='audio features')
parser.add_argument('--nb_epoch', type=int, default=300, help='number of epoch')
parser.add_argument('--margin', type=float, default=1.0, help='margin of the hinge loss')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--dir_order_train', type=str, default='data/train_order.h5', help='indices of training samples')
parser.add_argument('--dir_order_val', type=str, default='data/val_order.h5', help='indices of validation samples')
parser.add_argument('--dir_order_test', type=str, default='data/test_order.h5', help='indices of testing samples')
args = parser.parse_args()
args.data_root_path = '/media/wyk/DATA/datasets/AVE' if args.local else '/home2/wyk/datasets/AVE'
args.dir_video = args.data_root_path + '/' + args.dir_video
args.dir_audio = args.data_root_path + '/' + args.dir_audio

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:%i' % args.gpu) if torch.cuda.is_available() else torch.device('cpu')

model_name = args.name
net_model = Att_Net()

hinge_loss = nn.MarginRankingLoss(margin=args.margin)
optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=15000, gamma=0.1)


def train(args):
    AVEData = AVEDataset(dir_video=args.dir_video, dir_audio=args.dir_audio, 
                         dir_order=args.dir_order_train, batch_size=args.batch_size)
    nb_batch = len(AVEData) // args.batch_size
    epoch_l = []
    best_val_acc = 0
    print('Start training.')
    for epoch in range(args.nb_epoch):
        epoch_loss = 0
        n = 0
        start = time.time()
        for i in range(nb_batch):
            input_video, input_audio = AVEData.get_batch(i)
            input_video = input_video.to(device)
            input_audio_pos = input_audio.to(device)
            input_audio_neg = AVEData.neg_sampling().to(device)

            net_model.zero_grad()
            pos_norm = net_model(input_video, input_audio_pos)
            neg_norm = net_model(input_video, input_audio_neg)
            target = torch.ones_like(pos_norm)
            loss = hinge_loss(pos_norm, neg_norm, target)
            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            scheduler.step()
            optimizer.step()
            n += 1

        end = time.time()
        epoch_l.append(epoch_loss)
        print('=== Epoch {%s}  Loss: {%.4f}  Running time: {%2f}' % (str(epoch), (epoch_loss) / n, end - start))
        torch.save(net_model, 'model/' + model_name + ".pt")


train(args)
