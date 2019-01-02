'''AVE dataset'''
import numpy as np
import torch
import h5py

class AVEDataset(object):

    def __init__(self, video_dir, audio_dir, label_dir, order_dir, batch_size):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size

        with h5py.File(order_dir, 'r') as hf:
            order = hf['order'][:]
        self.lis = order

        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]  # [len, 10, 7, 7, 512]
        with h5py.File(video_dir, 'r') as hf:
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

        return torch.Tensor(self.audio_batch).float(), torch.Tensor(self.video_batch).float(), \
            torch.Tensor(self.label_batch).float()
