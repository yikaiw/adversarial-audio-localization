import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cf


class Attention(nn.Module):
    def __init__(self, att_method, score_method):
        super(Attention, self).__init__()
        self.att_method = att_method
        self.score_method = score_method
        hidden_size_1 = 512
        hidden_size_2 = 49 if att_method == 'add' else 128

        self.video_att_net = nn.Sequential(
            nn.Linear(512, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        )
        self.audio_att_net = nn.Sequential(
            nn.Linear(128, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        )
        self.att_affine = nn.Linear(hidden_size_2, 1, bias=False)
        self.video_embed_net = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        self.audio_embed_net = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        if score_method == 'concat':
            self.score_net = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1, bias=False))

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def init_weights(self):
        nn.init.xavier_uniform(self.video_att_net[0].weight)
        nn.init.xavier_uniform(self.video_att_net[2].weight)
        nn.init.xavier_uniform(self.audio_att_net[0].weight)
        nn.init.xavier_uniform(self.audio_att_net[2].weight)
        nn.init.xavier_uniform(self.att_affine.weight)
        nn.init.xavier_uniform(self.video_embed_net[0].weight)
        nn.init.xavier_uniform(self.video_embed_net[2].weight)
        nn.init.xavier_uniform(self.audio_embed_net[0].weight)
        nn.init.xavier_uniform(self.audio_embed_net[2].weight)
        if self.score_method == 'concat':
            nn.init.xavier_uniform(self.score_net[0].weight)
            nn.init.xavier_uniform(self.score_net[2].weight)

    def forward(self, video_input, audio_input):
        # video_input: [batch_size, 7, 7, 512], audio_input: [batch_size, 128]
        video_embed = video_input.view(-1, 49, 512)  # [batch_size, 49, 512]
        audio_embed = audio_input.view(-1, 128)

        # audio-guided visual attention
        video_att_embed = self.video_att_net(video_embed)  # [batch_size, 49, hidden_size_2]
        audio_att_embed = self.audio_att_net(audio_embed)  # [batch_size, hidden_size_2]

        if self.att_method == 'add':
            content = video_att_embed + audio_att_embed.unsqueeze(2)
            # [batch_size, 49, 49] = [batch_size, 49, 49] + [batch_size, 49, 1]
        elif self.att_method == 'cos':
            audio_att_embed = torch.cat([audio_att_embed.unsqueeze(1)] * 49, dim=1)
            content = torch.mul(video_att_embed, audio_att_embed)  # [batch_size, 49, hidden_size_2]

        e = self.att_affine((F.tanh(content))).squeeze(2)  # [batch_size, 49]
        alpha = F.softmax(e, dim=-1).unsqueeze(1)  # [batch_size, 1, 49]
        video_embed = torch.bmm(alpha, video_embed).view(-1, 512)  # [batch_size, 512]

        video_embed = self.video_embed_net(video_embed)  # [batch_size, 128]
        audio_embed = self.audio_embed_net(audio_embed)  # [batch_size, 128]
        if self.score_method == 'norm':
            sample_score = torch.norm(video_embed - audio_embed, dim=1)  # [batch_size]
            return sample_score  # less -> positive
        elif self.score_method == 'concat':
            sample_embed = torch.cat([video_embed, audio_embed], dim=1)  # [batch_size, 256]
            sample_score = self.score_net(sample_embed).squeeze()  # [batch_size]
            return sample_score  # less -> negative


class Discriminator(nn.Module):
    def __init__(self, margin):
        super(Discriminator, self).__init__()
        self.att = Attention(cf.att_method, cf.score_method)
        if cf.dis_loss == 'hinge':
            self.hinge_loss = nn.MarginRankingLoss(margin)

    def cal_reward(self, video_input, audio_neg_input):
        score = self.att(video_input, audio_neg_input)
        return score

    def forward(self, video_input, audio_pos_input, audio_neg_input):
        # [batch_size, 7, 7, 512], [batch_size, 128], [batch_size, 128]
        pos_score = self.att(video_input, audio_pos_input)
        neg_score = self.att(video_input, audio_neg_input)
        if cf.dis_loss == 'hinge':
            target = torch.ones_like(pos_score)
            loss = self.hinge_loss(pos_score, neg_score, target)
        elif cf.dis_loss == 'ratio':
            tmp = [torch.exp(pos_score), torch.exp(neg_score)]
            pos_d, neg_d = tmp / torch.sum(tmp)
            loss = torch.mean(torch.norm(pos_d) + torch.norm(1 - neg_d))
        return loss


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.att = Attention(cf.att_method, cf.score_method)

    def forward(self, video_input, audio_neg_cands, dis):
        # [batch_size, 7, 7, 512], [batch_size, cand_size, 128]
        cand_size = audio_neg_cands.size()[1]
        video_input = torch.cat([video_input] * cand_size, dim=0)  # [batch_size * cand_size, 7, 7, 512]
        audio_neg_cands = audio_neg_cands.view(-1, 128)  # [batch_size * cand_size, 128]
        cand_scores = self.att(video_input, audio_neg_cands).view(-1, cand_size)  # [batch_size, cand_size]
        cand_probs = torch.softmax(cand_scores, dim=1)  # [batch_size, cand_size]

        audio_neg_idx = utils.sample_from_probs(cand_probs.data.cpu().numpy())  # [batch_size]
        audio_neg_idx = torch.LongTensor(audio_neg_idx).unsqueeze(1)  # [batch_size, 1]
        audio_neg_prob = torch.gather(cand_probs, dim=1, index=audio_neg_idx).squeeze()  # [batch_size]

        audio_neg_cands = audio_neg_cands.view(-1, cand_size, 128)  # [batch_size, cand_size, 128]
        audio_neg_idx = torch.cat([audio_neg_idx.unsqueeze(2)] * 128, dim=2)  # [batch_size, 1, 128]
        audio_neg_input = torch.gather(audio_neg_cands, dim=1, index=audio_neg_idx).squeeze()  # [batch_size, 128]

        reward = dis.cal_reward(video_input, audio_neg_input)  # [batch_size]
        loss = -torch.mean(torch.mul(reward, torch.log(audio_neg_prob + 1e-5)))
        return loss
