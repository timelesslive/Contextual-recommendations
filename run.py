import json
import torch
import numpy as np
import random
import pprint
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from DIN import DeepInterestNetwork
from DIN import dataloader
from AFM import AttentionalFactorizationMachine
from transformers import BertTokenizer, BertModel ,logging
import os

device = torch.device("cuda")          
# 填充的函数 ：集体填充函数        
def pad_sequence_list(seq_list, max_len):
    return [seq + [0]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in seq_list]

def prePipeLine():
    '''
    Params:
    input : 从json文件中
    output : din_afm_input 的 dataframe
    获得最初始的数据,存储到datarame中
    '''
    
    ## step1 :  输入din的第一部分：用户交互历史处理 
    max_seq_length = 20 
    with open('user_info.json', 'r', encoding='utf-8') as file:
        user_data = json.load(file)
    df = pd.DataFrame(user_data)
    din_afm_input = pd.DataFrame()
    history = df['Browse']+df['Renewal']+df['Collection']+df['Reservation']
    din_afm_input['curBook'] = history.apply(lambda x: x[-2] if len(x) > 1 else None)
    din_afm_input['positiveSample'] = history.apply(lambda x: x[-1] if len(x) > 0 else None)
    filterHistory = history.apply(lambda x: x[:-2] if len(x) > 2 else [])
    concatResult = pad_sequence_list(filterHistory, max_seq_length)
    din_afm_input['bookHistory'] = concatResult
    # print(din_afm_input)
    
    # step2: 情景化特征：连续值的处理（时长相关）
    # 1. BrowserTime : 停留时间，以秒计时 。能表示用户当前总体的借书欲望 。为用户浏览的最后一本书的停留时长，以及最后一本的打开时间
    # 需要归一化到 [0, 1] 范围
    scaler_browser = MinMaxScaler()
    BrowserTime_norm = scaler_browser.fit_transform(df[['BrowserTime']])
    din_afm_input['BrowserTime'] = BrowserTime_norm.flatten()

    # 2. Datetime : 用户浏览的绝对时间,每天是一个循环，用分钟表示单日时间
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    Hour = df['Datetime'].dt.hour
    Minute = df['Datetime'].dt.minute
    times = Hour* 60 + Minute
    din_afm_input['Datetime'] = times / 1440.0
    # print(din_afm_input)
    
    # step3: 情景化特征：离散值的处理
    # 处理 'Month' 特征
    # 将 1-12 月映射为 0-11
    din_afm_input['Month'] = df['Month'] - 1

    # 处理 'Weekdays' 特征
    # 将 '周一' 到 '周日' 映射为 0-6
    weekdays_mapping = {
        '周一': 0,
        '周二': 1,
        '周三': 2,
        '周四': 3,
        '周五': 4,
        '周六': 5,
        '周日': 6
    }
    din_afm_input['Weekdays'] = df['Weekdays'].map(weekdays_mapping)

    # 处理 'Parttime' 特征
    # 将 '清晨'、'上午'、'中午'、'下午'、'夜晚' 映射为 0-4
    parttime_mapping = {
        '清晨': 0,
        '上午': 1,
        '中午': 2,
        '下午': 3,
        '夜晚': 4
    }
    din_afm_input['Parttime'] = df['Parttime'].map(parttime_mapping)

    # 处理 'PosType' 特征
    # 将 '文学'、'历史'、'小说'、'科技'、'其他' 映射为 0-4
    postype_mapping = {
        '文学': 0,
        '历史': 1,
        '小说': 2,
        '科技': 3,
        '其他': 4
    }
    din_afm_input['PosType'] = df['PosType'].map(postype_mapping)

    # 处理 'Weather' 特征
    # 将天气类型映射为 0-6
    weather_mapping = {
        '晴': 0,
        '多云': 1,
        '雨': 2,
        '雪': 3,
        '冰雹': 4,
        '霜冻': 5,
        '雾': 6
    }
    din_afm_input['Weather'] = df['Weather'].map(weather_mapping)

    # 处理'city'特征
    city_mapping = {'北京': 0, '上海': 1, '广州': 2, '深圳': 3, '杭州': 4, '成都': 5, '重庆': 6, '西安': 7, '南京': 8, '苏州': 9, '武汉': 10} 
    din_afm_input['City'] = df['City'].apply(lambda x:x.strip()).map(city_mapping)

    # 处理 'Search' 特征
    din_afm_input['Search'] = df['Search']

    # 处理 'Temp' 特征
    # 将温度等级映射为 0-3
    def temp_mapping(temprature):
        return int(temprature / 10)    
    din_afm_input['Temp'] = df['Temp'].map(temp_mapping)

    # 处理 'Humidity' 特征
    # 将湿度值映射为 0-4
    def encode_humidity(humidity):
        return int(humidity / 20)
    din_afm_input['Humidity'] = df['Humidity'].apply(encode_humidity)

    # 处理 'Windscale' 特征
    # 将风级区间映射为 0-3
    def windscale_mapping(windscale):
        return int(windscale / 4)
    din_afm_input['Windscale'] = df['Windscale'].map(windscale_mapping)

    # 处理 'Noise' 特征
    # 将噪音等级映射为 0-4
    def noise_mapping(noise):
        return int(noise / 20)

    din_afm_input['Noise'] = df['Noise'].map(noise_mapping)

    # step4 : 用户特征，user profile features
    din_afm_input['UserId'] = df['UserId'] 
    
    din_afm_input['Name']= df['Name']
    
    gender_mapping = {'男性': 0, '女性': 1}
    din_afm_input['Gender'] = df['Gender'].map(gender_mapping)
    
    def age_mapping(age):
        return int(age / 5) # 0-100岁
    din_afm_input['Age'] = df['Age'].map(age_mapping)
    
    education_mapping = {'小学': 0, '初中': 1, '高中': 2, '本科': 3, '硕士': 4, '博士': 5, '其他': 6}
    din_afm_input['Education'] = df['Education'].map(education_mapping)

    major_mapping = {'计算机科学与技术': 0, '机械设计制造及其自动化': 1, '软件工程': 2, '市场营销': 3, '电气工程': 4, '自动化': 5, '能源与动力': 6, '环境工程': 7, '电子信息': 8, '法学': 9, '临床医学': 10, '汉语言文学': 11, '其他': 12}
    din_afm_input['Major'] = df['Major'].map(major_mapping)
    
    marital_mapping = {'未婚': 0, '已婚': 1, '离异': 2, '丧偶': 3}
    din_afm_input['Marital'] = df['Marital'].map(marital_mapping)
    

    interest_mapping = {'运动': 0, '音乐': 1, '棋类': 2, '阅读': 3, '旅行': 4, '游戏': 5, '电影': 6, '其他': 7}
    din_afm_input['Interest'] = df['Interest'].map(interest_mapping)
    
    # print(din_afm_input.head())
    print(din_afm_input.columns)
    
    return din_afm_input 


def getUserFeature(df):
    '''
    params:
    input : 从 dataframe
    output : user_feature数组
    得到的dataframe转换为输入模型的数组
    '''
    # 读取book_info 
    with open('book_info.json', 'r', encoding='utf-8') as file:
        book_info = json.load(file)
    with open('DIN/model_config.json', 'r', encoding='utf-8') as file:
        dim_config = json.load(file)

    user_features = []
    for idx, row in df.iterrows():
        # 1. user_id 对应于 UserId
        user_feature = {
            'user_id': None,
            'user_gender': None,
            'user_age': None,
            'user_education': None,
            'user_major': None,
            'user_marital': None,
            'user_interest': None,
            
            'history_article_id': [],
            'history_text_feature': [],
            'history_categories': [],
            'query_article_id_candidatelist': [],
            'query_text_feature_candidatelist': [],
            'query_categories_candidatelist': [],
            'query_article_id': None,
            'query_text_feature': None,
            'query_categories': None,
            'positivesample': None,
            'positivesample_text_feature': None,
            'positivesample_categories': None,
            
            'situation_search': None,
            'situation_curbook': None,
            'situation_browsertime': None,
            'situation_datetime': None,
            'situation_month': None,
            'situation_weekdays': None,
            'situation_parttime': None,
            'situation_postype': None,
            'situation_weather': None,
            'situation_city': None,
            'situation_temp': None,
            'situation_humidity': None,
            'situation_windscale': None,
            'situation_noise': None,
            
        }
        user_feature['user_id'] = row['UserId']
        
        # 2. user_gender 对应于 Gender 
        user_feature['user_gender'] = row['Gender']
        
        # 3. user_age 对应于 Age
        user_feature['user_age'] = row['Age']
        
        # 4. user_education 对应于 Education
        user_feature['user_education'] = row['Education'] 
        
        # 5. user_major 对应于 Major
        user_feature['user_major'] = row['Major'] 
        
        # 6. user_marital 对应于 Marital
        user_feature['user_marital'] = row['Marital'] 
        
        # 7. user_interest 对应于 Interest
        user_feature['user_interest'] = row['Interest']  
        
        # 8. history_article_id 从 bookHistory 获取 。 由于模型输入的需要，要保持 0 的位置作为填充
        book_history = [x for x in row['bookHistory']]
        user_feature['history_len'] = len(book_history)
        user_feature['history_article_id'] = book_history

        # 9. history_text_feature：根据 history_article_id 获取 title
        history_text_feature = [book_info[str(book_id)]['title'] for book_id in book_history]
        user_feature['history_text_feature'] = history_text_feature
        
        # 10. history_categories：根据 history_article_id 获取 type
        history_categories = [book_info[str(book_id)]['type'] for book_id in book_history]
        user_feature['history_categories'] = history_categories
        
        # 11. query_article_id：从 dim_config 内挑选3个id，且不包含 curBook 和 positiveSample 。表示负样本
        valid_ids = list(set(range(1,dim_config['history_article_id'])) - {row['curBook'], row['positiveSample']})
        query_article_ids = random.sample(valid_ids, 3)
        user_feature['query_article_id_candidatelist'] = query_article_ids
        
        # 12. query_text_feature：根据 query_article_id 获取 title
        query_text_feature = [book_info[str(book_id)]['title'] for book_id in query_article_ids]
        user_feature['query_text_feature_candidatelist'] = query_text_feature
        
        # 13. query_categories：根据 query_article_id 获取 type
        query_categories = [book_info[str(book_id)]['type'] for book_id in query_article_ids]
        user_feature['query_categories_candidatelist'] = query_categories
        user_features.append(user_feature)
        
        # 正样本 
        user_feature['positivesample'] = row['positiveSample']
        user_feature['positivesample_text_feature'] = book_info[str(row['positiveSample'])]['title']
        user_feature['positivesample_categories'] = book_info[str(row['positiveSample'])]['type']

        # 进一步读入情景化数据
        # situation_curbook 对应于 curBook
        user_feature['situation_search'] = row['Search']
        
        user_feature['situation_curbook'] = row['curBook']
        user_feature['situation_browsertime'] = row['BrowserTime']
        user_feature['situation_datetime'] = row['Datetime']
        user_feature['situation_month'] = row['Month']
        user_feature['situation_weekdays'] = row['Weekdays']
        user_feature['situation_parttime'] = row['Parttime']
        user_feature['situation_postype'] = row['PosType']
        user_feature['situation_weather'] = row['Weather']
        user_feature['situation_city'] = row['City']
        user_feature['situation_temp'] = row['Temp']
        user_feature['situation_humidity'] = row['Humidity']
        user_feature['situation_windscale'] = row['Windscale']
        user_feature['situation_noise'] = row['Noise']
        
        
        '''
        转换成tensor
        1. 离散的 LongTensor list
        2. 连续的 FloatTensor list
        3. 单个的 LongTensor
        4. text字段: bert-base-chinese 进行编码
        5. None字段: 为query部分,放在后续选择性处理
        处理 非text 、 None 的字段 : 单个的 LongTensor,单个的 FloatTensor
        '''
        current_path = os.path.abspath(os.path.dirname(__file__))
        cache_dir = os.path.join(current_path, 'transformers')
        logging.set_verbosity_error()
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',cache_dir=cache_dir)
        bertmodel = BertModel.from_pretrained('bert-base-chinese',cache_dir=cache_dir)
        dim_bert = 768
        for key in user_feature: 
            if user_feature[key] is None:
                continue
            elif 'text' in key or 'search' in key:
                if isinstance(user_feature[key], list):  # 说明是文章标题列表，不会很长，所以就给16个token的固定长度
                    for i in range(len(user_feature[key])):
                        text = user_feature[key][i]
                        words_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=16)
                        bert_embedding = bertmodel(**words_input).pooler_output.detach()  # BERT 默认的句子表示是 768 维的
                        user_feature[key][i] = bert_embedding
                    user_feature[key] = torch.stack(user_feature[key]).squeeze(-2).to(device)
                else: 
                    # search检索的内容，默认不会很长，也给16个token的固定长度
                    text = user_feature[key]
                    words_input  = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=16)
                    bert_embedding = bertmodel(**words_input).pooler_output.detach()
                    user_feature[key] = torch.FloatTensor(bert_embedding).squeeze(-2).to(device)
            elif isinstance(user_feature[key], list): # 说明是个LongTensor 列表
                user_feature[key] = np.array(user_feature[key])
                user_feature[key] = torch.LongTensor(user_feature[key]).unsqueeze(1).to(device)
            elif isinstance(user_feature[key], int): # 说明是个LongTensor
                user_feature[key] = torch.LongTensor([user_feature[key]]).to(device)
            else:
                user_feature[key] = torch.FloatTensor([user_feature[key]]).to(device)
            # print('key :', key)
            # print('values size :' , user_feature[key].size())
    return user_features


# TODO 
# 1. 从 prompt 结果中拿出代码   √
# 2. 跑通模型√
# 3. 在模型中加入情景化部分的输入  √
# 4. 跑通模型  √
# 5. 加入 bert-base-chinese 预训练模型 ，对title进行编码  √
# 6. 跑通模型  √
# 7. 替换 user profile features 为 AFM 模型 
# 8. 加入训练部分的代码 ，划分测试集和训练集，搞清楚损失函数和负样本的处理
# 9. 跑通模型
# 10. 加入测试部分的代码
# 11. 跑通模型
if __name__ == "__main__":
    # step 1 : 读入数据
    din_afm_input = prePipeLine()
    print('real fields show\n', din_afm_input.columns)
    user_features = getUserFeature(din_afm_input)
    
   
    with open('DIN/model_config.json', 'r', encoding='utf-8') as file:
        dim_config = json.load(file)
    model = DeepInterestNetwork(dim_config)
    batch_size = dim_config['batch_size']
    
    
     # step 2 : 输入示例数据先把模型跑通 
    user_features_exsample = {
        # 用户表征
        'user_id': torch.LongTensor(np.ones(shape=(batch_size, 1))),  
        'user_gender': torch.LongTensor(np.ones(shape=(batch_size, 1))),
        'user_age': torch.LongTensor(np.ones(shape=(batch_size, 1))),  
        'user_education': torch.LongTensor(np.ones(shape=(batch_size, 1))), 
        'user_major': torch.LongTensor(np.ones(shape=(batch_size, 1))),  
        'user_marital': torch.LongTensor(np.ones(shape=(batch_size, 1))),  
        'user_interest': torch.LongTensor(np.ones(shape=(batch_size, 1))),  
        'history_len': torch.LongTensor(np.ones(shape=(batch_size, 1)) * 10),
        # 情景化特征
        'situation_curbook': torch.LongTensor(np.ones(shape=(batch_size, 1))),
        'situation_browsertime': torch.FloatTensor(np.ones(shape=(batch_size, 1))),  # 停留时间 ，Floattensor -> linear layer
        'situation_datetime': torch.FloatTensor(np.ones(shape=(batch_size, 1))),  # 时间，Floattensor -> linear layer
        
        'situation_search': torch.FloatTensor(np.random.rand(batch_size, 768)),
        
        'situation_month': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 月份
        'situation_weekdays': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 星期
        'situation_parttime': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 时间段
        'situation_postype': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 类型
        'situation_weather': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 天气
        'situation_city': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 城市
        'situation_temp': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 温度
        'situation_humidity': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 湿度
        'situation_windscale': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 风级
        'situation_noise': torch.LongTensor(np.ones(shape=(batch_size, 1))),  # 噪音
        # 实际标签
        'positivesample': torch.LongTensor(np.ones(shape=(batch_size, 1))),
    }
    # 用户历史行为特征
    user_features_exsample['history_article_id'] =  torch.LongTensor(np.random.randint(1, 17, size=(batch_size, 8,1)))  # 假设有 8 个历史交互，每个书籍ID在 [0, 16) 范围内 。这里的8是根据数据集的内容来定的，上限则是history_len
    user_features_exsample['history_text_feature'] = torch.FloatTensor(np.random.rand(batch_size, 8, 768))  # 每个历史交互的文本特征
    user_features_exsample['history_categories'] = torch.LongTensor(np.random.randint(0, 3, size=(batch_size, 8,1)))  # 每个历史交互的类别

    # 当前候选物品特征
    user_features_exsample['query_article_id'] = torch.LongTensor(np.random.randint(1, 17, size=(batch_size, 1)))  # 候选书籍的ID
    user_features_exsample['query_text_feature'] = torch.FloatTensor(np.random.rand(batch_size, 768))  # 候选书籍的文本特征
    user_features_exsample['query_categories'] = torch.LongTensor(np.random.randint(0, 3, size=(batch_size, 1)))  # 候选书籍的类别
    
    for key in user_features_exsample:
        user_features_exsample[key] = user_features_exsample[key].to(device)
    
    # step3 : 实际输入的数据
    '''    
    这一步的输入 
    1.离散特征没有被embedding  
    2.连续特征没有被 linear   
    3.文本特征已经被 bert 编码但是没有被 fc 降维到 text_embedding = 64 
    4.全部都已经是放到gpu上的tensor
    '''
    # TODO : 训练的正样本和三个负样本 .每次都需要手动挑选一个 query_article_id 。 否则只赋值了 *_candidatelist 属性
    for realsample in user_features:
        candidate_book_list =[realsample['positivesample'], 
                        realsample['query_article_id_candidatelist'][0], 
                        realsample['query_article_id_candidatelist'][1], 
                        realsample['query_article_id_candidatelist'][2]]
        # 作为样例输入，这里默认拿正样本跑通
        realsample['query_article_id'] = realsample['positivesample']
        realsample['query_text_feature'] = realsample['positivesample_text_feature']
        realsample['query_categories'] = realsample['positivesample_categories']

    # step4 : 推理跑通过程
    for i  in range(0,len(user_features), batch_size):
        batch_samples = user_features[i:i + batch_size]
        batch_input = dataloader.batchify(batch_samples, batch_size)
        output = model(batch_input)
        break

    # step5 : 模型训练
    # ctr 预估的本质是给定一个物品的各个方面的特征，直接根据这个特征输出一个 0-1 的概率值，表示这个物品被点击的概率
    # 在情景化推荐场景下，输入除了这个物品的特征值之外，还包括： 历史序列 + 用户特征 + 情景化特征。但本质上还是根据这些全部的特征输出一个 0-1 的概率值

   

    










