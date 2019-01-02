import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class att_Net(nn.Module):
    '''Audio-visual event localization with audio-guided visual attention and audio-visual fusion'''
    def __init__(self, hidden_dim, hidden_size, tagset_size):
        super(att_Net, self).__init__()
        self.hidden_dim = hidden_dim

        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, hidden_size)
        self.affine_video = nn.Linear(512, hidden_size)
        self.affine_v = nn.Linear(hidden_size, 49, bias=False)
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)
        self.affine_h = nn.Linear(49, 1, bias=False)

        self.affine_dec1 = nn.Linear(512, 256)
        self.affine_dec2 = nn.Linear(256, 128)

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def init_weights(self):
        '''initialize the weights'''
        nn.init.xavier_uniform(self.affine_audio.weight)
        nn.init.xavier_uniform(self.affine_video.weight)
        nn.init.xavier_uniform(self.affine_v.weight)
        nn.init.xavier_uniform(self.affine_g.weight)
        nn.init.xavier_uniform(self.affine_h.weight)
        nn.init.xavier_uniform(self.affine_dec1.weight)
        nn.init.xavier_uniform(self.affine_dec2.weight)

    def forward(self, audio, video):
        # audio: [batch size, 128], video: [batch size, 7, 7, 512]
        embed_video = video.view(video.size(0), -1, 512)  # [batch size, 49, 512]
        embed_video_tmp = embed_video

        # audio-guided visual attention
        embed_video = self.relu(self.affine_video(embed_video))  # [batch size, 49, hidden size]
        embed_audio = self.relu(self.affine_audio(audio))  # [batch size, hidden size]
        content_v = self.affine_v(embed_video) + self.affine_g(embed_audio).unsqueeze(2)
        # [batch size, 49, 49] = [batch size, 49, 49] + [batch size, 49, ]
        z = self.affine_h((F.tanh(content_v))).squeeze(2)  # [batch size, 49]
        alpha = F.softmax(z, dim=-1).unsqueeze(1)  # [batch size, 1, 49]
        embed_video = torch.bmm(alpha, embed_video_tmp).view(-1, 512)  # [batch size, 512]

        # video to audio
        embed_audio = self.affine_dec2(self.relu(self.affine_dec1(embed_video)))  # [batch size, 512]

        return embed_audio