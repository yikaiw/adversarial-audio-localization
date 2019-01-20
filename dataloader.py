'''AVE dataset'''
import numpy as np
import torch
import h5py
import config as cf

class AVEDataset(object):
    def __init__(self, video_dir, audio_dir, order_dir, batch_size):
        self.batch_size = batch_size

        with h5py.File(order_dir, 'r') as hf:
            order = hf['order'][:]  # numpy.ndarray: [3339]
        with h5py.File(video_dir, 'r') as hf:
            self.video_feature = hf['avadataset'][:]  # numpy.ndarray: [4143, 10, 7, 7, 512]
        with h5py.File(audio_dir, 'r') as hf:
            self.audio_feature = hf['avadataset'][:]  # numpy.ndarray: [4143, 10, 128]

        self.video_feature = self.video_feature[order]  # [3339, 10, 7, 7, 512]
        self.audio_feature = self.audio_feature[order]  # [3339, 10, 128]

        self.len = len(order) * 10  # 3339 * 10
        permu = np.arange(self.len)
        self.video_feature = np.reshape(self.video_feature, [-1, 7, 7, 512])[permu]  # [3339 * 10, 7, 7, 512]
        self.audio_feature = np.reshape(self.audio_feature, [-1, 128])[permu]  # [3339 * 10, 128]

    def __len__(self):
        return self.len

    def shuffle(self):
        permu = np.random.permutation(self.len)
        self.video_feature = self.video_feature[permu]
        self.audio_feature = self.audio_feature[permu]

    def get_batch(self, idx):
        video_batch = self.video_feature[idx * self.batch_size: (idx + 1) * self.batch_size]
        audio_batch = self.audio_feature[idx * self.batch_size:(idx + 1) * self.batch_size]
        # [batch_size, 7, 7, 512], [batch_size, 128]
        return torch.Tensor(video_batch).float().to(cf.device), torch.Tensor(audio_batch).float().to(cf.device)

    def neg_sampling(self, cand_size=1):
        idx = np.random.randint(self.len, size=(self.batch_size, cand_size)).squeeze()
        audio_batch_cands = self.audio_feature[idx]  # [batch_size, cand_size, 128]
        return torch.Tensor(audio_batch_cands).float().to(cf.device)
