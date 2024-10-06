import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

device = torch.device("cuda")
def batchify(samples, batch_size):
    batch = {}
    keys = samples[0].keys()
    for key in keys:
        feature_list = []
        for realsample in samples:
            feature_list.append(realsample[key])
        batch[key] = torch.stack(feature_list, dim=0)
    return batch



def custom_collate_fn(batch):
    # batch 是一个列表，包含多个 (data_row, label)
    data_list, label_list = zip(*batch)  
    labels = torch.tensor(label_list, dtype=torch.float32).to(device) # 转换为张量

    batch_dict = {}
    # 获取所有字段的名称
    keys = data_list[0].keys()
    print('keys:',keys)
    for key in keys:
        field_values = [sample[key] for sample in data_list]
        field_tensor = torch.stack(field_values, dim=0)
        batch_dict[key] = field_tensor
        print('batch_dict[key].size():',field_tensor.size())
    print('type:',type(batch_dict))
    print('batch_dict.keys():',batch_dict.keys())
    
    if(len(list(label_list)) == 1):
        labels = labels.unsqueeze(0)
    print('labels.size():',labels.size())
    return batch_dict, labels

class CustomDataset(Dataset):
    def __init__(self, data_df, labels):
        self.data = data_df.reset_index(drop=True)
        self.labels = labels
        assert len(self.data) == len(self.labels), "Data and labels must have the same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据行，转为字典
        data_row = self.data.iloc[idx].to_dict()
        label = self.labels[idx]
        return data_row, label
