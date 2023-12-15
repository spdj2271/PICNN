import torch
import numpy as np
from util.utils import AverageMeter, ProgressMeter
import time


def classspecific_train(train_loader, model, criterion, optimizer, epoch, logger, p):
    loss_total = AverageMeter('loss_total', ':.3f')
    loss_interpretation = AverageMeter('loss_interpretation', ':.3f')
    loss_discrimination = AverageMeter('loss_discrimination', ':.3f')
    n_positive_samples = AverageMeter('N_posi_sample', ':.0f')
    ACC1 = AverageMeter('ACC1', ':.5f')
    ACC2 = AverageMeter('ACC2', ':.5f')
    ACC3 = AverageMeter('ACC3', ':.5f')
    progress = ProgressMeter(len(train_loader),
                             [loss_total, loss_discrimination, loss_interpretation, n_positive_samples, ACC1, ACC2,
                              ACC3], logger, prefix="Epoch: [{}]".format(epoch))

    
    model.train()
    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(images, targets=targets, forward_pass='train')
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss['loss_total'].backward()
        optimizer.step()

        loss_total.update(loss['loss_total'])
        loss_interpretation.update(loss['loss_interpretation'])
        loss_discrimination.update(loss['loss_discrimination'])
        n_positive_samples.update(loss['n_positive_sample'])
        ACC1.update(loss['ACC1'])
        ACC2.update(loss['ACC2'])
        ACC3.update(loss['ACC3'])

        if i % 50 == 0:
            progress.display(i)
            
    return loss_total.avg
