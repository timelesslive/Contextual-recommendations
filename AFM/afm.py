# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:41:39 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn

class AttentionalFactorizationMachine(nn.Module):
    def __init__(self, feature_fields, embed_dim, attn_size, dropouts):
        super(AttentionalFactorizationMachine, self).__init__()
        print('=============into init==================')
        self.num_fields = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        print('embedding input dim:',sum(feature_fields)+1)
        #线性部分
        self.linear = nn.Embedding(sum(feature_fields)+1, 1)
        self.bias =nn.Parameter(torch.zeros((1,)))  
        
        #embedding
        self.embedding = nn.Embedding(sum(feature_fields)+1, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight.data)
        
        #attention部分
        self.attention = nn.Linear(embed_dim, attn_size)
        self.projection = nn.Linear(attn_size, 1)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropouts = dropouts
    
    def forward(self, x):
        print('=============into forward==================')
        print('input dim:',x.shape)
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0) #构造成embeding形式
        linear_part = torch.sum(self.linear(tmp), dim = 1) + self.bias # 线性部分
        
        tmp = self.embedding(tmp) # embedding后的vec
        
        # 交叉项, 并加入attention
        num_fields = tmp.shape[1]
        row, col = [], []
        for i in range(num_fields - 1):
            for j in range(i+1, num_fields):
                row.append(i)
                col.append(j)
        p, q = tmp[:, row], tmp[:,col]
        inner = p * q
        attn_scores = nn.functional.relu(self.attention(inner))
        attn_scores = nn.functional.softmax(self.projection(attn_scores), dim=1)
        attn_scores = nn.functional.dropout(attn_scores, p = self.dropouts[0])
        attn_output = torch.sum(attn_scores * inner, dim = 1)
        attn_output = nn.functional.dropout(attn_output, p = self.dropouts[1])
        print('attn_output dim:',attn_output.shape)
        inner_attn_part = self.fc(attn_output)
        print('Linear part dim:',linear_part.shape)
        print('inner_attn_part dim:',inner_attn_part.shape)
        # 最后输出
        x = linear_part + inner_attn_part
        print('afm out dim:',x.shape)
        x = torch.sigmoid(x.squeeze(1))
        print('afm out dim after sigmoid :',x.shape)
        print('sample out:',x[:5])
        return x