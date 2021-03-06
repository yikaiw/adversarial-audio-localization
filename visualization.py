from __future__ import print_function
import os
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from dataloader import AVEDatasetSeq
import imageio
import cv2
import warnings
warnings.filterwarnings('ignore')
import argparse
import config as cf


parser = argparse.ArgumentParser(description='Visualization')
# Data specifications
parser.add_argument('--note', type=str, default=None, help='model note')
parser.add_argument('--local', action='store_true', default=False, help='run locally or remotely')
parser.add_argument('--save_origin', action='store_true', default=False, help='save origin images')
parser.add_argument('--gpu', type=int, default=0, help='gpu selection')
parser.add_argument('--batch_size', type=int, default=100, help='select the number of the results')
args = parser.parse_args()
args.data_root_path = cf.data_root_path_local if args.local else cf.data_root_path_remote
args.save_root_path = cf.save_root_path_local if args.local else cf.save_root_path_remote

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cf.device = torch.device('cuda:%i' % args.gpu) if torch.cuda.is_available() else torch.device('cpu')

def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):
        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))
    return num


def normlize(x, min = 0, max = 255):
    num, row, col = x.shape
    for i in range(num):
        xi = x[i, :, :]
        xi = max * (xi - np.min(xi)) / (np.max(xi) - np.min(xi))
        x[i, :, :] = xi
    return x


# features, and testing set list
video_dir = '%s/visual_feature.h5' % args.data_root_path
audio_dir = '%s/audio_feature.h5' % args.data_root_path
order_dir_test = 'data/test_order.h5'

# access to original videos for extracting video frames
raw_video_dir = '%s/AVE_Dataset/AVE' % args.data_root_path  # videos in AVE dataset
# lis = os.listdir(raw_video_dir)
f = open('data/Annotations.txt', 'r')
dataset = f.readlines()
print('The dataset contains %d samples' % (len(dataset)))
f.close()
len_data = len(dataset)
with h5py.File(order_dir_test, 'r') as hf:
    test_order = hf['order'][:]

# pre-trained models
if torch.cuda.is_available():
    att_model = torch.load('model/AV_att.pt')
else:
    att_model = torch.load('model/AV_att.pt', map_location='cpu')
att_layer = att_model._modules.get('att_affine')  # extract attention maps from the layer

# load testing set
AVEData = AVEDatasetSeq(video_dir=video_dir, audio_dir=audio_dir,
                        order_dir=order_dir_test, batch_size=args.batch_size)
nb_batch = len(AVEData)
print('number of batch: %i' % nb_batch)
video_input, audio_input = AVEData.get_batch(1)

# generate attention maps
att_map = torch.zeros(args.batch_size * 10, 49, 1)

def fun(m, i, o):
    att_map.copy_(o.data)

map = att_layer.register_forward_hook(fun)
h_x = att_model(video_input, audio_input)
map.remove()
z_t = Variable(att_map.squeeze(2))
alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))
att_weight = alpha_t.view(args.batch_size, 10, 7, 7).cpu().data.numpy()
# attention maps of all testing samples
    
c = 0
t = 10
sample_num = 16  # 16 frames for 1-sec video segment
extract_frames = np.zeros((160, 224, 224, 3))  # 160 224x224x3 frames for a 10-sec video
subfolder = 'att-%s' % cf.time if args.note is None else 'att-%s-%s' % (cf.time, args.note)
save_dir = '%s/%s/' % (args.save_root_path, subfolder)  # store attention maps
original_dir = '%s/%s/original' % (args.save_root_path, subfolder)  # store video frames

for num in range(args.batch_size):
    print(num)
    data = dataset[test_order[num]]
    x = data.split('&')
    
    # extract video frames
    video_index = os.path.join(raw_video_dir, x[1] + '.mp4')
    vid = imageio.get_reader(video_index, 'ffmpeg')
    vid_len = len(vid)
    frame_interval = int(vid_len / t)
  
    frame_num = video_frame_sample(frame_interval, t, sample_num)
    imgs = []
    for i, im in enumerate(vid):
        x_im = cv2.resize(im, (224, 224))
        imgs.append(x_im)
    vid.close()
    cc = 0
    for n in frame_num:
        extract_frames[cc, :, :, :] = imgs[n]
        cc += 1
    
    # process generated attention maps
    att = att_weight[num, :, :, :]
    att = normlize(att, 0, 255)
    att_scaled = np.zeros((10, 224, 224))
    for k in range(att.shape[0]):
        att_scaled[k, :, :] = cv2.resize(att[k, :, :], (224, 224))  # scaling attention maps 
  
    att_t = np.repeat(att_scaled, 16, axis=0)
    # 1-sec segment only has 1 attention map. Here, repeat 16 times to generate 16 maps for a 1-sec video
    heat_maps = np.repeat(att_t.reshape(160, 224, 224, 1), 3, axis = -1)
    c += 1
    
    att_dir = save_dir + str(num)
    ori_dir =  original_dir + str(num)
    if not os.path.exists(att_dir):
        os.makedirs(att_dir)
    if args.save_origin and not os.path.exists(ori_dir):
        os.makedirs(ori_dir)
    for idx in range(160):
        heat_map = heat_maps[idx, :, :, 0]
        im = extract_frames[idx, :, :, :]
        im = im[:, :, (2, 1, 0)]
        heatmap = cv2.applyColorMap(np.uint8(heat_map), cv2.COLORMAP_JET)
        
        att_frame = heatmap * 0.2 + np.uint8(im) * 0.6
        n = '%04d' % idx
        vid_index = os.path.join(att_dir, 'pic' + n + '.jpg')
        cv2.imwrite(vid_index, att_frame)
        if args.save_origin:
            ori_frame = np.uint8(im)
            ori_index = os.path.join(ori_dir, 'ori' + n + '.jpg')
            cv2.imwrite(ori_index, ori_frame)

ip = '166.111.138.137' if args.local else '166.111.138.222'
print('scp -r wyk@%s:%s' % (ip, save_dir))
