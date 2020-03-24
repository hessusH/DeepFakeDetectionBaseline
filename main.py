import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import yaml
from models import classificators
import numpy as np
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import datetime


def train(model, optimizer, scheduler, criterion, train_loader, epoch, writer, config, device):
    model.train()
    model.to(device)
    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backwarconfigd()
        #             avg.append(loss.item())
        optimizer.step()
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)

    pass
# def val():
#     pass
# def test():
#     pass


def main(config):
    date = datetime.datetime.now()
    writer = SummaryWriter(log_dir=str(date) + '_' + config['model_name'])
    model = getattr(classificators, config['model_name'])(config['num_classes'])
    writer.add_graph(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    scheduler = getattr(optim.lr_scheduler, config['lr_scheduler'])(optimizer, config['step'])
    train_loader = None
    test_loader = None
    val_loader = None
    for epoch in range(config['num_epochs']):
        train(model, optimizer, scheduler, criterion, train_loader, epoch, writer, config['device'])
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, default='./config/default_config.yaml', help='Choose config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    main(config)