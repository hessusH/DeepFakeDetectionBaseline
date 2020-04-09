import yaml
import datetime
import argparse
import numpy as np

import torch
import torch.optim as optim
import torchvision
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

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

    for i, (images, labels) in enumerate(train_loader):


        images = images.to(device)
        labels = labels.to(device)
        output = model(images).view(-1)
        loss = criterion(output, labels)
        loss.backward()

        if (i + 1) % config['step'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        pred = torch.sigmoid(output) > 0.5
        labels = labels > 0.5
        
        accuracy = metrics.accuracy(pred, labels)
        loss_handler.update(loss)
        accuracy_handler.update(accuracy)
        writer.add_scalar('Train/Loss', loss_handler.avg, epoch * len(train_loader) + i)
        writer.add_scalar('Train/Accuracy', accuracy_handler.avg, epoch * len(train_loader) + i)


def validation(val_loader, model, criterion, thresholds, device):
    model.eval()
    
    loss_handler = AverageMeter()
    accuracy_handler = [AverageMeter() for _ in thresholds]
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images).view(-1)
            
            loss = criterion(output, labels)
            loss_handler.update(loss)
    
            labels = labels.byte()
            for i, threshold in enumerate(thresholds):
                
                pred = torch.sigmoid(output) > threshold
                accuracy = metrics.accuracy(pred, labels)
                accuracy_handler[i].update(accuracy)
                
    return (loss_handler.avg, [i.avg for i in accuracy_handler])



def main(config):
    device = torch.device('cuda:0' if (config['device'] == 'gpu' and torch.cuda.is_available()) else 'cpu')
    model = getattr(classificators, config['model_name'])()
    model = torchvision.models.resnet50(num_classes = config['num_classes'])
    model.to(device)

    if config['snapshot']['use']:
        load_weights(model, config['prefix'], config['model']['name'])
        start_epoch = config['snapshot']['epoch']
    else:
        start_epoch = 0

    if torch.cuda.is_available() and config['parallel']:
        model = DataParallel(model)

    date = datetime.datetime.now()
    writer = SummaryWriter()

    writer.add_graph(model, torch.rand([1, 3, 512, 512]).to(device))
    criterion = getattr(losses, config['loss'])()
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.5,
                                                     patience=2,
                                                     min_lr=1e-6)
    train_loader = get_deepfake_train(config)
    val_loader = get_deepfake_val(config)

    best_loss = np.inf
    tresholds = np.linspace(config['tresholds']['start'], config['tresholds']['finish'], config['tresholds']['steps'])
    for epoch in range(start_epoch, config['num_epochs']):
        train(model, optimizer, criterion, train_loader, epoch, writer, config, device)
        
        loss, accuracy = validation(val_loader, model, criterion, tresholds, device)
        writer.add_scalar('Val/Loss', loss, epoch)
        writer.add_scalar('Val/Accuracy', max(accuracy), epoch)

        if best_loss > loss:
            best_loss = loss
            save_weights(model, config['prefix'], config['model_name'], f'best{epoch}', config['parallel'])
        
        if epoch != 0:
            scheduler.step(loss)
            
        if epoch % config['save_frequency'] == 0:
            save_weights(model, config['prefix'], config['model_name'], epoch, config['parallel'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, default='config/default_config.yaml', help='Choose config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    main(config)
