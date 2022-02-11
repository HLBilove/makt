# Code reused from https://github.com/arghosh/AKT

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class MAKT(nn.Module):
    def __init__(self, n_skill, n_exercise, n_if, n_a1, n_a2, dataset, d_model, n_blocks, kq_same, dropout, 
                model_type, final_fc_dim=512, n_heads=8, d_ff=2048, l2=1e-5, epsilon=1e-5):
        super().__init__()
        self.n_skill = n_skill
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_exercise = n_exercise
        self.n_if = n_if
        self.n_a1 = n_a1
        self.n_a2 = n_a2
        self.dataset = dataset
        self.l2 = l2
        self.model_type = model_type
        self.log_vars = nn.Parameter(torch.zeros((3)))
        self.epsilon = epsilon
        embed_l = d_model
        concat_embed_l = embed_l

        if self.n_exercise > 0:
            self.difficult_param = nn.Embedding(self.n_exercise + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_skill+1, embed_l)
            self.qa11_embed_diff = nn.Embedding(2*self.n_skill+1, embed_l)
            self.qa12_embed_diff = nn.Embedding(self.n_a1*self.n_skill+1, embed_l)
            self.qa21_embed_diff = nn.Embedding(2*self.n_skill+1, embed_l)
            self.qa22_embed_diff = nn.Embedding(self.n_a2*self.n_skill+1, embed_l)
        else:
            self.q_embed_diff = nn.Embedding(self.n_exercise+1, embed_l)
            self.qa11_embed_diff = nn.Embedding(2, embed_l)
            self.qa12_embed_diff = nn.Embedding(self.n_a1, embed_l)
            self.qa21_embed_diff = nn.Embedding(2, embed_l)
            self.qa22_embed_diff = nn.Embedding(self.n_a2, embed_l)
            concat_embed_l += embed_l

        # n_skill+1 ,d_model
        self.q_embed = nn.Embedding(self.n_skill+1, embed_l)
        self.qa_embed = nn.Embedding(2, embed_l)
        self.qa1_embed = nn.Embedding(self.n_a1, embed_l)
        self.qa2_embed = nn.Embedding(self.n_a2, embed_l)

        self.a11_sam_param = nn.Embedding(self.n_skill+2, 1)
        self.a12_sam_param = nn.Embedding(self.n_skill+2, 1)
        self.a21_sam_param = nn.Embedding(self.n_skill+2, 1)
        self.a22_sam_param = nn.Embedding(self.n_skill+2, 1)
        self.iam_param = nn.Embedding(self.n_if+1, 1)
        self.iam_bias_param = nn.Embedding(self.n_if+1, embed_l)

        self.a1_sam_param = nn.Embedding(self.n_skill+3, 1)
        self.a2_sam_param = nn.Embedding(self.n_skill+3, 1)

        self.x_softmax = nn.Sequential(nn.Linear(concat_embed_l, concat_embed_l), nn.Softmax(1), nn.Dropout(self.dropout))
        self.y1_softmax = nn.Sequential(nn.Linear(concat_embed_l, concat_embed_l), nn.Softmax(1), nn.Dropout(self.dropout))
        self.y2_softmax = nn.Sequential(nn.Linear(concat_embed_l, concat_embed_l), nn.Softmax(1), nn.Dropout(self.dropout))

        self.output1_softmax = nn.Sequential(nn.Linear(concat_embed_l, concat_embed_l), nn.Softmax(1), nn.Dropout(self.dropout))
        self.output2_softmax = nn.Sequential(nn.Linear(concat_embed_l, concat_embed_l), nn.Softmax(1), nn.Dropout(self.dropout))
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_skill=n_skill, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,
                                kq_same=self.kq_same, model_type=self.model_type)

        d_model = concat_embed_l
        self.out_a = nn.Sequential(
            nn.Linear(d_model + d_model + d_model,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        self.out_a1 = nn.Sequential(
            nn.Linear(d_model + d_model + d_model,
                      final_fc_dim), nn.Softmax(1), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.Softmax(1), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        self.out_a2 = nn.Sequential(
            nn.Linear(d_model + d_model + d_model,
                      final_fc_dim), nn.Softmax(1), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.Softmax(1), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_skill+2 and self.n_skill > 0:
                torch.nn.init.constant_(p, 0.5)
            if p.size(0) == self.n_skill+3 and self.n_skill > 0:
                torch.nn.init.constant_(p, 1.)
            if p.size(0) == self.n_exercise+1 and self.n_exercise > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_if+1 and self.n_if > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, qa_data, qa1_data, qa2_data, target_a, target_a1, target_a2, e_data=None, if_data=None):
        # Batch First
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        a11_weight = self.a11_sam_param(q_data)
        a12_weight = self.a12_sam_param(q_data)
        a21_weight = self.a21_sam_param(q_data)
        a22_weight = self.a22_sam_param(q_data)
        iam_weight = self.iam_param(if_data)
        iam_bias = self.iam_bias_param(if_data)

        a_data = (qa_data-q_data)//self.n_skill  # rt
        a1_data = (qa1_data-q_data)//self.n_skill
        a2_data = (qa2_data-q_data)//self.n_skill

        qa1_embed_data = a11_weight*self.qa_embed(a_data)+a12_weight * self.qa1_embed(a1_data)+self.q_embed(q_data)
        qa2_embed_data = a21_weight*self.qa_embed(a_data)+a22_weight * self.qa2_embed(a2_data)+self.q_embed(q_data)

        x_embed = q_embed_data
        y1_embed = qa1_embed_data
        y2_embed = qa2_embed_data

        if self.n_exercise > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            e_embed_data = self.difficult_param(e_data)
            x_embed += iam_weight * e_embed_data * q_embed_diff_data + iam_bias

            pa1_embed_data = iam_weight * e_embed_data * (a11_weight * self.qa11_embed_diff(a_data)+a12_weight*self.qa12_embed_diff(a1_data)+q_embed_diff_data) + iam_bias
            pa2_embed_data = iam_weight * e_embed_data * (a21_weight * self.qa21_embed_diff(a_data)+a22_weight*self.qa22_embed_diff(a2_data)+q_embed_diff_data) + iam_bias
            y1_embed += pa1_embed_data
            y2_embed += pa2_embed_data

        x_embed = self.x_softmax(x_embed)
        y1_embed = self.y1_softmax(y1_embed)
        y2_embed = self.y2_softmax(y2_embed)

        d1_output = self.model(x_embed, y1_embed)
        d2_output = self.model(x_embed, y2_embed)

        a1_weight = self.a1_sam_param(q_data)
        a2_weight = self.a2_sam_param(q_data)

        if self.dataset == 'assist2009':
            d1_output = a1_weight * self.output1_softmax(d1_output)
            d2_output = a2_weight * self.output2_softmax(d2_output)
        else:
            d1_output = a1_weight * d1_output
            d2_output = a2_weight * d2_output

        concat_q = torch.cat([d1_output, d2_output, x_embed], dim=-1)

        a_output = self.out_a(concat_q)
        a1_output = self.out_a1(concat_q)
        a2_output = self.out_a2(concat_q)

        a_labels = target_a.reshape(-1)
        a1_labels = target_a1.reshape(-1)
        a2_labels = target_a2.reshape(-1)

        m = nn.Sigmoid()

        a_preds = (a_output.reshape(-1))
        a1_preds = (a1_output.reshape(-1))
        a2_preds = (a2_output.reshape(-1))
        a_mask = a_labels > -0.9
        a1_mask = a1_labels > -0.9
        a2_mask = a2_labels > -0.9
        masked_a_labels = a_labels[a_mask].float()
        masked_a_preds = a_preds[a_mask]
        masked_a1_labels = a1_labels[a1_mask].float()
        masked_a1_preds = a1_preds[a1_mask]
        masked_a2_labels = a2_labels[a2_mask].float()
        masked_a2_preds = a2_preds[a2_mask]
       
        #UWLoss
        m1 = nn.Sigmoid()
        masked_a_preds = m1(masked_a_preds)
        masked_a1_preds = m1(masked_a1_preds)
        masked_a2_preds = m1(masked_a2_preds)
        loss1 = -torch.mean((1-masked_a_preds) * masked_a_labels * torch.log(masked_a_preds + self.epsilon) + 
        masked_a_preds * (1-masked_a_labels) * torch.log(1-masked_a_preds + self.epsilon))
        loss2 = -torch.mean((1-masked_a1_preds) * masked_a1_labels * torch.log(masked_a1_preds + self.epsilon) + 
        masked_a1_preds* (1-masked_a1_labels) * torch.log(1-masked_a1_preds + self.epsilon))
        loss3 = -torch.mean((1-masked_a2_preds) * masked_a2_labels * torch.log(masked_a2_preds + self.epsilon) + 
        masked_a2_preds * (1-masked_a2_labels) * torch.log(1-masked_a2_preds + self.epsilon))
        output = loss1 / (2 * torch.exp(self.log_vars[0])) + self.log_vars[0]
        output += loss2 / (2 * torch.exp(self.log_vars[1])) + self.log_vars[1]
        output += loss3 / (2 * torch.exp(self.log_vars[2])) + self.log_vars[2]

        return output.sum(), m(a_preds), a_mask.sum(), m(a1_preds), a1_mask.sum(), m(a2_preds), a2_mask.sum()

class Architecture(nn.Module):
    def __init__(self, n_skill,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'makt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
