import os
import torch


def load_weights(model, prefix, model_name, epoch):
    file = os.path.join('./snapshots', f'{prefix}_{model_name}_epoch_{epoch}.pth')
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['state_dict'])


def save_weights(model, prefix, model_name, epoch, parallel=True):
    file = os.path.join('./snapshots', f'{prefix}_{model_name}_epoch_{epoch}.pth')
    if torch.cuda.is_available() and parallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'state_dict': state_dict}, file)
