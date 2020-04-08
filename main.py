import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
import argparse
import yaml
from models import classificators
from utils.storage import load_weights, save_weights
import numpy as np
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import datetime





def train(model, optimizer, scheduler, criterion, train_loader, epoch, writer, config, device):
    model.train()
    model.to(device)
    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backwarconfigd()
        optimizer.step()
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)

    pass
# def val():
#     pass
# def test():
#     pass


def main(config):
    device = torch.device('cuda:0' if (config['device'] == 'gpu' and torch.cuda.is_available()) else 'cpu')


    if config['snapshot']['use']:
        model = getattr(classificators, config['model_name'])(config['num_classes'])
        model.to(device)
        load_weights(model, config['prefix'], config['model']['name'], config['snapshot']['epoch'])
    else:
        model = getattr(classificators, config['model_name'])(config['num_classes'])
        model.to(device)

    if torch.cuda.is_available() and config['parallel']:
        model = DataParallel(model)

    date = datetime.datetime.now()
    writer = SummaryWriter(log_dir=str(date) + '_' + config['model_name'])


    writer.add_graph(model)
    criterion = getattr(nn, config['criterion'])()
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    # scheduler = getattr(optim.lr_scheduler, config['lr_scheduler'])(optimizer, config['step'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=config['learning_rate']['decay'],
                                                     patience=config['learning_rate']['no_improve'],
                                                     min_lr=config['learning_rate']['min_val'])
    train_loader = None
    test_loader = None
    val_loader = None
    for epoch in range(config['num_epochs']):
        train(model, optimizer, scheduler, criterion, train_loader, epoch, writer)
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, default='./config/default_config.yaml', help='Choose config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    main(config)
