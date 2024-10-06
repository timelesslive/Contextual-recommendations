import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from afm import AttentionalFactorizationMachine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv('../data/data.csv')
data_X = data.iloc[:,2:]
data_y = data.click.values
data_X = data_X.apply(LabelEncoder().fit_transform)


fields = data_X.max().values + 1 # 模型输入的feature_fields
#train, validation, test 集合
tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)


# 数据量小, 可以直接读
train_X = torch.from_numpy(train_X.values).long()
val_X = torch.from_numpy(val_X.values).long()
test_X = torch.from_numpy(test_X.values).long()

train_y = torch.from_numpy(train_y).long()
val_y = torch.from_numpy(val_y).long()
test_y = torch.from_numpy(test_y).long()

train_set = Data.TensorDataset(train_X, train_y)
val_set = Data.TensorDataset(val_X, val_y)
train_loader = Data.DataLoader(dataset=train_set,
                               batch_size=32,
                               shuffle=True)
val_loader = Data.DataLoader(dataset=val_set,
                             batch_size=32,
                             shuffle=False)
epoches = 1
print(type(fields))
print(fields.shape)
model = AttentionalFactorizationMachine(feature_fields=fields, embed_dim=8, attn_size=8, dropouts=(0.25, 0.25))
out  = model(train_X)
# _ = train(model)
# input size :(batch_size , input_dim)


