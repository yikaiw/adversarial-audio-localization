'''AVE dataset'''
import numpy as np
import torch
import h5py

class AVEDataset(object):

    def __init__(self, dir_video, dir_audio, dir_order, batch_size):
        self.dir_video = dir_video
        self.dir_audio = dir_audio
        self.batch_size = batch_size

        with h5py.File(dir_order, 'r') as hf:
            order = hf['order'][:]
        self.lis = order

        with h5py.File(dir_audio, 'r') as hf:
            self.audio_features = hf['avadataset'][:]  # [len, 10, 7, 7, 512]
        with h5py.File(dir_video, 'r') as hf:
            self.video_features = hf['avadataset'][:]  # [len, 10, 128]

        permu = np.random.permutation(len(self.audio_features) * 10)
        self.audio_features = np.reshape(self.audio_features, [-1, 7, 7, 512])[permu]  # [len * 10, 7, 7, 512]
        self.video_features = np.reshape(self.video_features, [-1, 128])[permu]  # [len * 10, 128]

        self.video_batch = np.zeros([self.batch_size, 7, 7, 512], dtype=np.float32)
        self.audio_batch = np.zeros([self.batch_size, 128], dtype=np.float32)

    def __len__(self):
        return len(self.lis)

    def get_batch(self, idx):
        for i in range(self.batch_size):
            id = idx * self.batch_size + i
            self.video_batch[i, :, :, :] = self.video_features[self.lis[id], :, :, :]
            self.audio_batch[i, :] = self.audio_features[self.lis[id], :]

        return torch.Tensor(self.video_batch).float(), torch.Tensor(self.audio_batch).float()
