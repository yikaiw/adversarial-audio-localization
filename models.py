import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cf


class Attention_Net(nn.Module):
    '''Audio-visual event localization with audio-guided visual attention and audio-visual fusion'''
    def __init__(self):
        super(Attention_Net, self).__init__()
        self.att_type = 'add'  # add, cos
        hidden_size_1 = 512
        hidden_size_2 = 49 if self.att_type == 'add' else 128

        self.net_att_video = nn.Sequential(
            nn.Linear(512, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        )
        self.net_att_audio = nn.Sequential(
            nn.Linear(128, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        )
        self.affine_att = nn.Linear(hidden_size_2, 1, bias=False)
        self.net_embed_video = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        self.net_embed_audio = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        if cf.score_method == 'concat':
            self.score_net = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1, bias=False))

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def init_weights(self):
        nn.init.xavier_uniform(self.net_att_video[0].weight)
        nn.init.xavier_uniform(self.net_att_video[2].weight)
        nn.init.xavier_uniform(self.net_att_audio[0].weight)
        nn.init.xavier_uniform(self.net_att_audio[2].weight)
        nn.init.xavier_uniform(self.affine_att.weight)
        nn.init.xavier_uniform(self.net_embed_video[0].weight)
        nn.init.xavier_uniform(self.net_embed_video[2].weight)
        nn.init.xavier_uniform(self.net_embed_audio[0].weight)
        nn.init.xavier_uniform(self.net_embed_audio[2].weight)
        if cf.score_method == 'concat':
            nn.init.xavier_uniform(self.score_net[0].weight)
            nn.init.xavier_uniform(self.score_net[2].weight)

    def forward(self, input_video, input_audio):
        # input_video: [batch_size, 10, 7, 7, 512], input_audio: [batch_size, 10, 128]
        embed_video = input_video.view(-1, 49, 512)  # [batch_size * 10, 49, 512]
        embed_audio = input_audio.view(-1, 128)

        # audio-guided visual attention
        embed_att_video = self.net_att_video(embed_video)  # [batch_size * 10, 49, hidden_size_2]
        embed_att_audio = self.net_att_audio(embed_audio)  # [batch_size * 10, hidden_size_2]

        if self.att_type == 'add':
            content = embed_att_video + embed_att_audio.unsqueeze(2)
            # [batch_size * 10, 49, 49] = [batch_size * 10, 49, 49] + [batch_size * 10, 49, 1]
        elif self.att_type == 'cos':
            embed_att_audio = torch.cat([embed_att_audio.unsqueeze(1)] * 49, dim=1)
            content = torch.mul(embed_att_video, embed_att_audio)  # [batch_size * 10, 49, hidden_size_2]

        e = self.affine_att((F.tanh(content))).squeeze(2)  # [batch_size * 10, 49]
        alpha = F.softmax(e, dim=-1).unsqueeze(1)  # [batch_size * 10, 1, 49]
        embed_video = torch.bmm(alpha, embed_video).view(-1, 512)  # [batch_size * 10, 512]

        embed_video = self.net_embed_video(embed_video)  # [batch_size * 10, 128]
        embed_audio = self.net_embed_audio(embed_audio)  # [batch_size * 10, 128]
        if cf.score_method == 'norm':
            score_sample = torch.norm(embed_video - embed_audio, dim=1)  # [batch_size * 10, 1]
            return score_sample
        elif cf.score_method == 'concat':
            embed_sample = torch.cat([embed_video, embed_audio], dim=1)  # [batch_size * 10, 256]
            score_sample = self.score_net(embed_sample)  # [batch_size * 10, 1]
            return score_sample


class Discriminator(nn.Module):
    def __init__(self, margin):
        super(Discriminator, self).__init__()
        self.hinge_loss = nn.MarginRankingLoss(margin)

    def forward(self, score_pos, score_neg):
        target = torch.ones_like(score_pos)
        hinge_loss = self.hinge_loss(score_pos, score_neg, target)
        return hinge_loss


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def init_weights(self):
        pass
        
