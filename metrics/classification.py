import torch

def accuracy(output, target):
    with torch.no_grad():
        res = torch.sum(output == target)
        return res.float() / target.size(0)
