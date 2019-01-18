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
        self.lis = order  # numpy.ndarray: [3339, ]


        with h5py.File(dir_video, 'r') as hf:
            self.feature_video = hf['avadataset'][:]  # numpy.ndarray: [4143, 10, 7, 7, 512]
        with h5py.File(dir_audio, 'r') as hf:
            self.feature_audio = hf['avadataset'][:]  # numpy.ndarray: [4143, 10, 128]

        # permu = np.random.permutation(len(self.feature_video) * 10)
        # self.feature_video = np.reshape(self.feature_video, [-1, 7, 7, 512])[permu]  # [len * 10, 7, 7, 512]
        # self.feature_audio = np.reshape(self.feature_audio, [-1, 128])[permu]  # [len * 10, 128]

        self.batch_video = np.zeros([self.batch_size, 10, 7, 7, 512], dtype=np.float32)
        self.batch_audio = np.zeros([self.batch_size, 10, 128], dtype=np.float32)

    def __len__(self):
        return len(self.lis)

    def get_batch(self, idx):
        for i in range(self.batch_size):
            id = idx * self.batch_size + i
            self.batch_video[i] = self.feature_video[self.lis[id]]
            self.batch_audio[i] = self.feature_audio[self.lis[id]]
        return torch.Tensor(self.batch_video).float(), torch.Tensor(self.batch_audio).float()

    def neg_sampling(self, cs=1):
        batch_audio = []
        for i in range(self.batch_size):
            id = np.random.randint(len(self.lis), size=cs).squeeze()
            batch_audio.append(self.feature_audio.take(self.lis.take(id)))
        return torch.Tensor(batch_audio).float()
