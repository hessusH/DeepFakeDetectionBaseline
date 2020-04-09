import yaml
import datetime
import argparse
import numpy as np

import torch
import torch.optim as optim
import torchvision
from torch.nn import DataParallel
# from torch.utils.tensorboard import SummaryWriter

import losses
from utils.handlers import AverageMeter
import metrics.classification as metrics
from models import classificators
from utils.storage import load_weights, save_weights
from data.datasets.deep_fake import get_deepfake_train, get_deepfake_val





def train(model, optimizer, criterion, train_loader, epoch, writer, config, device):
    model.train()

    loss_handler = AverageMeter()
    accuracy_handler = AverageMeter()

    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        loss = criterion(output, label)
        loss.backward()

        if (i + 1) % config['step'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        pred = torch.sigmoid(output) > 0.5
        target = target > 0.5

        accuracy = metrics.accuracy(pred, target)
        loss_handler.update(loss)
        accuracy_handler.update(accuracy)
        print(loss, accuracy)
        # writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)

def val():
    pass



def main(config):
    device = torch.device('cuda:0' if (config['device'] == 'gpu' and torch.cuda.is_available()) else 'cpu')
    # model = getattr(classificators, config['model_name'])(config['num_classes'])
    model = torchvision.models.resnet18()
    model.to(device)

    if config['snapshot']['use']:
        load_weights(model, config['prefix'], config['model']['name'])
        start_epoch = config['snapshot']['epoch']
    else:
        start_epoch = 0

    if torch.cuda.is_available() and config['parallel']:
        model = DataParallel(model)

    date = datetime.datetime.now()
    writer = None #SummaryWriter(log_dir=str(date) + '_' + config['model_name'])

    # writer.add_graph(model)
    criterion = getattr(losses, config['loss'])()
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.5,
                                                     patience=2,
                                                     min_lr=1e-6)
    train_loader = get_deepfake_train(config)
    val_loader = get_deepfake_val(config)

    best_epoch = 0
    best_loss = np.inf

    for epoch in range(start_epoch, config['num_epochs']):
        train(model, optimizer, criterion, train_loader, epoch, writer, config, device)
        #
        # loss, accuracy = validation(val_loader, model, criterion)
        #
        # if best_loss > loss:
        #     best_loss = loss
        #     best_epoch = epoch + 1
        #     save_weights(model, config['prefix'], config['model_name'], f'best{best_epoch}', config['parallel'])
        #
        # if epoch != 0:
        #     scheduler.step(loss)
        #
        # save_weights(model, config['prefix'], config['model_name'], epoch + 1, config['parallel'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, default='./config/default_config.yaml', help='Choose config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    main(config)
