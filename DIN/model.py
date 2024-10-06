import torch.nn as nn
import torch
import numpy as np
from .embedding import EmbeddingLayer
from .fc import FullyConnectedLayer
from .attention import AttentionSequencePoolingLayer
from AFM import AttentionalFactorizationMachine
# from embedding import EmbeddingLayer
# from fc import FullyConnectedLayer
# from attention import AttentionSequencePoolingLayer
import json
import os

# user profile features
# 把所有可能需要用到的 多个维度的特征 ， 以"user_"开头放到这里面
device = torch.device("cuda")  
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'model_config.json')
with open(config_path, 'r', encoding='utf-8') as file:
    dim_config = json.load(file)
text_hidden_dim = 64
category_hidden_dim = 8 

# query 部分是 candidate item
que_embed_features = ['query_article_id']
que_text_features = ['query_text_feature']
que_category =  ['query_categories']

# his 部分是 user behavior
his_embed_features = ['history_article_id']
his_text_features = ['history_text_feature']
his_category =  ['history_categories']

embed_features = [k for k, _ in dim_config.items() if 'user' in k]
embed_context = [k for k, _ in dim_config.items() if 'situation' in k]

class DeepInterestNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embedding_dim = config['embedding_dim']
        context_dim = config['context_dim']
        
        # step 1: candidate item 部分的网络结构
        self.query_feature_embedding_dict = dict()
        # 书籍的id ||  feature_dim 表示书籍总数 
        for feature in que_embed_features: # que_embed_features ： ['query_article_id']
            self.query_feature_embedding_dict[feature] = EmbeddingLayer(feature_dim=dim_config[feature],
                                                                        embedding_dim=embedding_dim).to(device)
        # 模态id，这里使用 text_feature 。输入768维，输出64维
        self.query_text_fc = FullyConnectedLayer(input_size=768,
                                                  hidden_size=[text_hidden_dim],
                                                  bias=[True],
                                                  activation='relu').to(device)
        # 类别id
        self.query_cate_embedding = nn.Embedding(category_hidden_dim, embedding_dim).to(device)
        
        # step 2: User Behavior 部分的网络结构 。也是书籍id + text_feature + category
        self.history_feature_embedding_dict = dict()
        for feature in his_embed_features:
            self.history_feature_embedding_dict[feature] = EmbeddingLayer(feature_dim=dim_config[feature],
                                                                          embedding_dim=embedding_dim).to(device)    
        self.history_text_fc = FullyConnectedLayer(input_size=768,
                                                    hidden_size=[text_hidden_dim],
                                                    bias=[True],
                                                    activation='relu').to(device)                                                     
        # print('attention - embedding_dim',text_hidden_dim + embedding_dim + category_hidden_dim)  # dim : 80 = 64 + 8 + 8 ,for each good of user behavior
        self.attn = AttentionSequencePoolingLayer(embedding_dim=text_hidden_dim + embedding_dim + category_hidden_dim).to(device)

        # step3 : user profile features 部分的embedding层 
        # self.user_id_embedding = nn.Embedding(config['user_id'], embedding_dim).to(device)
        # self.user_gender_embedding = nn.Embedding(config['user_gender'], embedding_dim).to(device)
        # self.user_age_embedding = nn.Embedding(config['user_age'], embedding_dim).to(device)
        # self.user_education_embedding = nn.Embedding(config['user_education'], embedding_dim).to(device)
        # self.user_major_embedding = nn.Embedding(config['user_major'], embedding_dim).to(device)
        # self.user_marital_embedding = nn.Embedding(config['user_marital'], embedding_dim).to(device)
        # self.user_interest_embedding = nn.Embedding(config['user_interest'], embedding_dim).to(device)
        
        afm_inpit_dim = np.stack([dim_config[feature] for feature in embed_features])
        print('afm_inpit_dim.shape',afm_inpit_dim.shape)
        self.AttentionalFactorizationMachine = AttentionalFactorizationMachine(feature_fields=afm_inpit_dim,embed_dim=10*embedding_dim, attn_size=8, dropouts=(0.25, 0.25)).to(device)
        
        # : step4 : context feature 的embedding层
        self.situation_curbook_embedding = nn.Embedding(config['situation_curbook'], context_dim).to(device)
        self.sitation_browsertime_ln = nn.Linear(1, context_dim).to(device)
        self.sitation_datetime_ln = nn.Linear(1, context_dim).to(device)
        # 假设已经获得了bert-base-chinese的文本特征 ,这个特征是64维的
        self.situation_search_fc = FullyConnectedLayer(input_size=768,
                                            hidden_size=[text_hidden_dim],
                                            bias=[True],
                                            activation='relu').to(device)
        self.situation_month_embedding = nn.Embedding(config['situation_month'], context_dim).to(device)
        self.situation_weekdays_embedding = nn.Embedding(config['situation_weekdays'], context_dim).to(device)
        self.situation_parttime_embedding = nn.Embedding(config['situation_parttime'], context_dim).to(device)
        self.situation_postype_embedding = nn.Embedding(config['situation_postype'], context_dim).to(device)
        self.situation_weather_embedding = nn.Embedding(config['situation_weather'], context_dim).to(device)
        self.situation_city_embedding = nn.Embedding(config['situation_city'], context_dim).to(device)
        self.situation_temp_embedding = nn.Embedding(config['situation_temp'], context_dim).to(device)
        self.situation_humidity_embedding = nn.Embedding(config['situation_humidity'], context_dim).to(device)
        self.situation_windscale_embedding = nn.Embedding(config['situation_windscale'], context_dim).to(device)
        self.situation_noise_embedding = nn.Embedding(config['situation_noise'], context_dim).to(device)
        
        # map for step3 and step4
        self.mapping = {
            # 'user_id': self.user_id_embedding,
            # 'user_gender': self.user_gender_embedding,
            # 'user_age': self.user_age_embedding,
            # 'user_education': self.user_education_embedding,
            # 'user_major': self.user_major_embedding,
            # 'user_marital': self.user_marital_embedding,
            # 'user_interest': self.user_interest_embedding,
            'situation_curbook': self.situation_curbook_embedding,
            'situation_browsertime': self.sitation_browsertime_ln,
            'situation_datetime': self.sitation_datetime_ln,
            'situation_search': self.situation_search_fc,
            'situation_month': self.situation_month_embedding,
            'situation_weekdays': self.situation_weekdays_embedding,
            'situation_parttime': self.situation_parttime_embedding,
            'situation_postype': self.situation_postype_embedding,
            'situation_weather': self.situation_weather_embedding,
            'situation_city': self.situation_city_embedding,
            'situation_temp': self.situation_temp_embedding,
            'situation_humidity': self.situation_humidity_embedding,
            'situation_windscale': self.situation_windscale_embedding,
            'situation_noise': self.situation_noise_embedding
        }
        
        # step5 : book history + book candidate  + user profile + context feature
        self.fc_layer = FullyConnectedLayer(input_size=2 * (text_hidden_dim + embedding_dim + category_hidden_dim) + dim_config['afm_out_dim'] + (len(embed_context)-1)*context_dim+text_hidden_dim, 
                                            hidden_size=[200, 80, 1],
                                            bias=[True, True, False],
                                            activation='relu',
                                            sigmoid=True).to(device)

    def forward(self, user_features):
        # 先将输入的实际内容转换为embedding
        # user_features -> dict (key:feature name, value: feature tensor)
        
        # step1 : user profile 部分
        # deep input embedding
        user_feature_embedded = []

        # embed_features ： user profile features 各个特征的维度  feature_size  表示用户本身的特征有多少个，这里配置文件里面有7个 user_ 开头的特征
        
        for feature in embed_features:
            # TODO : 将 user_features[feature]堆叠之后传入到 afm 的输入中 ,维度要求是：batch_size * sum(feature_dim)
            print('user_features[feature].squeeze().size()',user_features[feature].squeeze().size())
            user_feature_embedded.append(user_features[feature].squeeze(-1))
            # embedding_layer = self.mapping.get(feature)  
            # embedded_feature = embedding_layer(user_features[feature].squeeze()) 
            # user_feature_embedded.append(embedded_feature)
            
        print('user_feature_embedded.len()',len(user_feature_embedded))  
        user_feature_embedded = torch.stack(user_feature_embedded, dim=1)
        print('user_feature_embedded.size()',user_feature_embedded.size())
        user_feature_out  = self.AttentionalFactorizationMachine(user_feature_embedded)
        # print('User_feature_embed size : ', user_feature_embedded.size()) # batch_size * (feature_size * embedding_dim) ，2 *（7 * 8）
        # print('User feature done')
        
        # step2: candidate item 部分
        query_feature_embedded = []

        for feature in que_embed_features:# que_embed_features ： ['query_article_id']
            # TODO:index out of range in self
            # print('index error place : user_features[feature]', user_features[feature])
            # print('index error place : dim.config :', dim_config[feature])
            id_embedding = self.query_feature_embedding_dict[feature](user_features[feature]).squeeze(1)
            # print('id_embedding size : ', id_embedding.size())
            query_feature_embedded.append(id_embedding)
        for feature in que_text_features:
            text_embedding = self.query_text_fc(user_features[feature])
            # print('text_embedding size : ', text_embedding.size())
            query_feature_embedded.append(text_embedding)
        for feature in que_category:
            category_embedding = self.query_cate_embedding(user_features[feature]).squeeze(1)
            # print('category_embedding size : ', category_embedding.size())
            query_feature_embedded.append(category_embedding)

        query_feature_embedded = torch.cat(query_feature_embedded, dim=1)
        # print('Query feature_embed size', query_feature_embedded.size()) # batch_size * (2 * embedding_dim + text_embedding_dim) ,2 *(2*8 + context_dim)
        # print('Query feature done')

        # step3 : history 部分
        history_feature_embedded = []
        for feature in his_embed_features: # his_embed_features ： ['history_article_id']
            user_history_embedding = self.history_feature_embedding_dict[feature](user_features[feature]).squeeze(-2)
            history_feature_embedded.append(user_history_embedding)
            print('user_history_embedding size : ', user_history_embedding.size())
        for feature in his_text_features: # his_text_features ： ['history_text_feature']
            history_text_embedding = self.history_text_fc(user_features[feature])
            history_feature_embedded.append(history_text_embedding)
            print('user_text_embedding size : ', history_text_embedding.size())
        for feature in his_category: # his_category ： ['history_categories']
            history_category_embedding = self.query_cate_embedding(user_features[feature]).squeeze(-2)
            history_feature_embedded.append(history_category_embedding)
            print('history_category_embedding size : ', history_category_embedding.size())

        history_feature_embedded = torch.cat(history_feature_embedded, dim=2)
        # print('History feature_embed size', history_feature_embedded.size()) # batch_size * T * (feature_size * embedding_dim) = 2 * 8 * (2 * 8 + context_dim)
        # print('History feature done')
        
        # print(user_features.keys())
        
        history = self.attn(query_feature_embedded.unsqueeze(1),   # (2,1,80)
                            history_feature_embedded,   # (2,8,80)
                            user_features['history_len']) # 进一步规定历史序列的长度
        
        # step4 : context feature 部分 . embedding + linear 层， 和 user profile feature 部分完全相同的处理
        context_feature_embedded = []
        for feature in embed_context:
            embedding_layer = self.mapping.get(feature)
            if isinstance(embedding_layer, nn.Embedding):
                embedded_feature = embedding_layer(user_features[feature].squeeze(-1))
                context_feature_embedded.append(embedded_feature)
            elif isinstance(embedding_layer, nn.Linear):
                embedded_feature = embedding_layer(user_features[feature])
                context_feature_embedded.append(embedded_feature)
        # 单独处理文本特征
        text_embedding = self.situation_search_fc(user_features['situation_search'])
        print('text_embedding size : ', text_embedding.size())
        context_feature_embedded.append(text_embedding)
        context_feature_embedded = torch.cat(context_feature_embedded, dim=1)
        
        # step5 : 合四个输入为一，四个输入加起来的维度是 fc 线形层初始化函数 的 input_size  。拼接之前统一不进行sigmoid
        print('user_feature_embedded.size()',user_feature_out.size())
        print('query_feature_embedded.size()',query_feature_embedded.size())
        print('history.size()',history.size())
        print('context_feature_embedded.size()',context_feature_embedded.size())
        concat_feature = torch.cat([user_feature_out, query_feature_embedded, history,context_feature_embedded], dim=1) # [2,80],[2,80],[2,80],[2,116]
        
        # fully-connected layers
        # print('concat_feature.size()',concat_feature.size())
        output = self.fc_layer(concat_feature) 
        print('model Inference Done !!')
        print('output_final.size()',output.size())
        return output


if __name__ == "__main__":
    batch_size = 2
    a = DeepInterestNetwork(dim_config)
    user_feature= {}
    a(user_feature)
    