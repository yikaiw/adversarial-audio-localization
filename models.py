import torch
import torch.nn as nn
import torch.nn.functional as F


class Att_Net(nn.Module):
    '''Audio-visual event localization with audio-guided visual attention and audio-visual fusion'''
    def __init__(self, hidden_size_1, hidden_size_2, noise_size=0, att_type='add'):
        super(att_Net, self).__init__()
        self.att_type = att_type
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, hidden_size_1)
        self.affine_video = nn.Linear(512, hidden_size_1)
        if att_type == 'add':
            hidden_size_2 = 49
        self.affine_v = nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        self.affine_g = nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        self.affine_h = nn.Linear(hidden_size_2, 1, bias=False)

        self.noise_size = noise_size
        self.decoder = nn.Sequential(nn.Linear(512 + noise_size, 256), nn.ReLU(), nn.Linear(256, 128))

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
        nn.init.xavier_uniform(self.decoder[0].weight)
        nn.init.xavier_uniform(self.decoder[2].weight)

    def forward(self, input_video, input_audio):
        # input_video: [batch_size, 10, 7, 7, 512], input_audio: [batch_size, 10, 128]
        embed_video = input_video.view(-1, 49, 512)  # [batch_size * 10, 49, 512]
        embed_video_tmp = embed_video

        # audio-guided visual attention
        embed_video = self.relu(self.affine_video(embed_video))  # [batch_size * 10, 49, hidden_size_1]
        embed_audio = input_audio.view(-1, 128)
        embed_audio = self.relu(self.affine_audio(embed_audio))  # [batch_size * 10, hidden_size_1]

        if self.att_type == 'add':
            content_v = self.affine_v(embed_video) + self.affine_g(embed_audio).unsqueeze(2)
            # [batch_size * 10, 49, 49] = [batch_size * 10, 49, 49] + [batch_size * 10, 49, 1]
        elif self.att_type == 'cos':
            embed_mul_video = self.affine_v(embed_video)
            embed_mul_audio = torch.cat([self.affine_g(embed_audio).unsqueeze(1)] * 49, dim=1)
            content_v = torch.mul(embed_mul_video, embed_mul_audio)  # [batch_size * 10, 49, hidden_size_2]
        e = self.affine_h((F.tanh(content_v))).squeeze(2)  # [batch_size * 10, 49]
        alpha = F.softmax(e, dim=-1).unsqueeze(1)  # [batch_size * 10, 1, 49]
        embed_video = torch.bmm(alpha, embed_video_tmp).view(-1, 512)  # [batch_size * 10, 512]

        z = torch.rand(embed_video.size()[0], self.noise_size, requires_grad=True).cuda()
        embed_video = self.decoder(torch.cat([embed_video, z], dim=-1))  # [batch_size * 10, 128]
        return torch.norm(embed_video - embed_audio)
