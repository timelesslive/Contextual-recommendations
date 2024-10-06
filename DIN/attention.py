import torch.nn as nn
import torch
import torch.nn.functional as F

from .fc import FullyConnectedLayer
# from fc import FullyConnectedLayer


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionSequencePoolingLayer, self).__init__()

        # TODO: DICE acitivation function
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(hidden_size=[64, 16], bias=[True, True], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, user_behavior_length):
        # query_ad : (B, 1, E)
        # user_behavior : (B, T, E)
        # user_behavior_length : (B, 1)
        # 输出 : (B, E)

        # 计算注意力得分
        attention_score = self.local_att(query_ad, user_behavior)  # (B, T, 1)
        attention_score = attention_score.squeeze(-1)  # (B, T)
        print('attention_score1.size()', attention_score.size())

        # 创建掩码
        device = user_behavior.device
        batch_size = user_behavior.size(0)
        max_len = user_behavior.size(1)
        user_behavior_length = user_behavior_length.squeeze(-1)  # (B,)
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < user_behavior_length.unsqueeze(1)  # (B, T)
        print('mask.size()', mask.size())

        # 应用掩码
        attention_score = attention_score.masked_fill(~mask, float('-inf'))
        print('attention_score2.size()', attention_score.size())

        # 归一化注意力得分
        attention_weights = F.softmax(attention_score, dim=-1)  # (B, T)
        print('attention_weights1.size()', attention_weights.size())

        # 调整维度以进行乘法
        attention_weights = attention_weights.unsqueeze(-1)  # (B, T, 1)
        print('attention_weights2.size()', attention_weights.size())

        # 计算加权和
        output = torch.sum(user_behavior * attention_weights, dim=1)  # (B, E)

        return output  # (B, E)

        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=8, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim, # 80 *4 
                                       hidden_size=hidden_size, # [64,16]
                                       bias=bias, # [True, True]
                                       batch_norm=batch_norm, # False
                                       activation='dice', 
                                       dice_dim=3)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)
        # TODO: fc_2 initialization

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)

        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior], dim=-1)
        print('attention_input.size()',attention_input.size())
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output

if __name__ == "__main__":
    a = AttentionSequencePoolingLayer()
    
    import torch
    b = torch.zeros((3, 1, 4))
    c = torch.zeros((3, 20, 4))
    d = torch.ones((3, 1))
    a(b, c, d)