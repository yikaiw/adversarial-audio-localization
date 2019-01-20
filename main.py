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
from dataloader import AVEDataset
import random
from models import Discriminator, Generator
random.seed(3344)
import time
import warnings
warnings.filterwarnings('ignore')
import argparse
import config as cf
import utils

parser = argparse.ArgumentParser(description='AVE')
# data specifications
parser.add_argument('--name', type=str, default='AV_att', help='model name')
parser.add_argument('--local', action='store_true', default=False, help='run locally or remotely')
parser.add_argument('--gpu', type=int, default=0, help='gpu selection')
parser.add_argument('--video_dir', type=str, default='visual_feature.h5', help='visual features')
parser.add_argument('--audio_dir', type=str, default='audio_feature.h5', help='audio features')
parser.add_argument('--epoch', type=int, default=300, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=1024, help='number of batch size')
parser.add_argument('--order_dir_train', type=str, default='data/train_order.h5', help='indices of training samples')
parser.add_argument('--order_dir_val', type=str, default='data/val_order.h5', help='indices of validation samples')
parser.add_argument('--order_dir_test', type=str, default='data/test_order.h5', help='indices of testing samples')
parser.add_argument('--margin', type=float, default=1.0, help='margin of the hinge loss')
parser.add_argument('--cand_size', type=int, default=1, help='candidate size for adversarial sampling')
args = parser.parse_args()
args.data_root_path = cf.data_root_path_local if args.local else cf.data_root_path_remote
args.video_dir = args.data_root_path + '/' + args.video_dir
args.audio_dir = args.data_root_path + '/' + args.audio_dir

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cf.device = torch.device('cuda:%i' % args.gpu) if torch.cuda.is_available() else torch.device('cpu')

model_name = args.name
dis = Discriminator(args.margin)
dis_optimizer = optim.Adam(dis.parameters(), lr=1e-3)
dis_scheduler = StepLR(dis_optimizer, step_size=15000, gamma=0.1)
if cf.sampling_method == 'adversarial':
    gen = Generator()
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3)
    gen_scheduler = StepLR(gen_optimizer, step_size=15000, gamma=0.1)


def train(args):
    AVEData = AVEDataset(video_dir=args.video_dir, audio_dir=args.audio_dir, 
                         order_dir=args.order_dir_train, batch_size=args.batch_size)
    step_num = len(AVEData) // args.batch_size
    dis_epoch_loss_list, gen_epoch_loss_list = [], []
    best_val_acc = 0
    print('Start training.')
    step = 0
    for epoch in range(args.epoch):
        dis_epoch_loss, gen_epoch_loss = 0, 0
        start = time.time()
        for i in range(step_num):
            video_input, audio_pos_input = AVEData.get_batch(i)
            # [batch_size, 7, 7, 512], [batch_size, 128]
            if cf.sampling_method == 'uniform':
                audio_neg_input = AVEData.neg_sampling()  # [batch_size, 128]
            elif cf.sampling_method == 'adversarial':
                audio_neg_cands = AVEData.neg_sampling(args.cand_size)  # [batch_size, cand_size, 128]
                gen.zero_grad()
                gen_loss = gen(video_input, audio_neg_cands, dis)
                gen_epoch_loss += gen_loss.data.cpu().numpy()
                gen_epoch_loss_list.append(gen_epoch_loss)
                dis_loss.backward()
                gen_scheduler.step()
                gen_optimizer.step()

            dis.zero_grad()
            dis_loss = dis(video_input, audio_pos_input, audio_neg_input)
            dis_epoch_loss += dis_loss.data.cpu().numpy()
            dis_loss.backward()
            dis_scheduler.step()
            dis_optimizer.step()
            step += 1

        end = time.time()
        dis_epoch_loss_list.append(dis_epoch_loss)
        if cf.sampling_method == 'uniform':
            print('Epoch {%s}  Loss: {%.4f}  Time: {%2f}'
                % (str(epoch), (dis_epoch_loss) / step_num, end - start))
        elif cf.sampling_method == 'adversarial':
            print('Epoch {%s}  Loss: {%.4f, %.4f}  Time: {%2f}'
                % (str(epoch), (dis_epoch_loss, gen_epoch_loss) / step_num, end - start))
    torch.save(att, 'model/' + model_name + '.pt')


train(args)
