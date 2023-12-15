import math
import numpy as np
import torch
import torchvision.transforms as transforms
from util.collate import collate_custom


def get_train_transformations(p, mean=None, std=None):
    if p['augmentation_strategy'] == 'standard':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p, mean=None, std=None):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_train_dataset(p, transform, to_augmented_dataset=False):

    if p['db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)
    
    else:
        raise ValueError('Invalid train dataset {}'.format(p['db_name']))

    return dataset


def get_val_dataset(p, transform=None, to_neighbors_dataset=False, n_picked_clusters=None):
    # Base dataset
    if p['db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)

    else:
        raise ValueError('Invalid validation dataset {}'.format(p['db_name']))

    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                       drop_last=False, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                       drop_last=False, shuffle=False)


def get_criterion(p):
    if p['criterion'] == 'ClassSpecificCE':
        from losses.losses import ClassSpecificCE
        criterion = ClassSpecificCE(**p['criterion_kwargs'])
    elif p['criterion'] == 'StandardCE':
        from losses.losses import StandardCE
        criterion = StandardCE()
    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_optimizer(p, model):
    params = model.parameters()
    if p['optimizer'] == 'sgd' or p['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))
    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

