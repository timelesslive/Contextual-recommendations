import torch

def batchify(samples, batch_size):
    batch = {}
    keys = samples[0].keys()
    for key in keys:
        feature_list = []
        for realsample in samples:
            feature_list.append(realsample[key])
        batch[key] = torch.stack(feature_list, dim=0)
    return batch
