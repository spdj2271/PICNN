import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from models.models import LoadModel
from util.common_config import get_train_transformations, get_val_transformations, get_train_dataset, \
    get_train_dataloader, get_val_dataloader, get_criterion, get_optimizer, adjust_learning_rate, \
    get_val_dataset
from util.config import create_config
from util.evaluate_utils import get_predictions, evaluate_ACC_MIS
from util.train_utils import classspecific_train
from util.utils import seed_torch
import time
import argparse


def main():
    seed_torch(0)
    cmd_opt = argparse.ArgumentParser(description='Argparser for PICNN')
    cmd_opt.add_argument('-configFileName', default='./configs/cifar10.yml')
    cmd_opt.add_argument('-criterion', default='ClassSpecificCE',help='StandardCE/ClassSpecificCE')
    cmd_opt.add_argument('-backbone', default='resnet18',help='resnet18')
    cmd_args, _ = cmd_opt.parse_known_args()
    

    p = create_config(config_file=cmd_args.configFileName,
                      backbone=cmd_args.backbone,
                      criterion=cmd_args.criterion)    

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(p['base_dir'], 'log.log')), logging.StreamHandler()])
    logger.info(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Model
    logging.info(colored('Get model', 'blue'))
    model = LoadModel(p)
    logger.info('Model is {}'.format(model.__class__.__name__))
    logger.info('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    model = model.cuda()
    transforms_backbone = model.weights.transforms()
    model = torch.nn.DataParallel(model)

    # Data
    logging.info(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p, mean=transforms_backbone.mean, std=transforms_backbone.std)
    base_transformations = get_val_transformations(p, mean=transforms_backbone.mean, std=transforms_backbone.std)
    train_dataset = get_train_dataset(p, train_transformations)
    val_dataset = get_val_dataset(p, base_transformations)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    logger.info('Train samples %d, val samples %d' % (len(train_dataset), len(val_dataset)))

    # Loss function
    logger.info(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    logger.info('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer
    logger.info(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    logger.info(optimizer)

    # Checkpoint
    if os.path.exists(p['best_checkpoint']):
        logger.info(colored('Restart from checkpoint {}'.format(p['best_checkpoint']), 'blue'))
        checkpoint = torch.load(p['best_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        logger.info(colored('No checkpoint file at {}'.format(p['best_checkpoint']), 'blue'))
        start_epoch = 0
        max_ACC_val = float('-inf')
    
    # Main loop
    logger.info(colored('Starting main loop', 'blue'))
    writer = SummaryWriter(f"{p['base_dir']}")
    train_time_total = 0.0
    val_time_total = 0.0
    for epoch in range(start_epoch, p['epochs']):
        print('epoch{}\n'.format(epoch))
        logger.info(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
        logger.info(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        logger.info('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        logger.info('Train ...')
        train_start_time = time.time()
        classspecific_train(train_dataloader, model, criterion, optimizer, epoch, logger, p)
        train_end_time = time.time()
        train_time_total += train_end_time-train_start_time
        
        # Evaluate
        logger.info('Make prediction on validation set ...')
        val_start_time = time.time()
        predictions = get_predictions(p, val_dataloader, model)
        val_end_time = time.time()
        val_time_total += val_end_time-val_start_time
        eval_stats = evaluate_ACC_MIS(predictions, model.module.correlation, epoch)
        logger.info(colored(eval_stats, 'green'))

        if eval_stats['ACC1'] > max_ACC_val:
            max_ACC_val = eval_stats['ACC1']
            best_stats = {'epoch_minloss': epoch + 1, 'ACC1': eval_stats['ACC1'],
                          'ACC2': eval_stats['ACC2'], 'ACC3': eval_stats['ACC3'],
                          'MIS': eval_stats['MIS']}
            logger.info(colored(f"New lowest loss: {best_stats}", 'red'))
            writer.add_scalar(tag='best_model/ACC1', scalar_value=best_stats['ACC1'], global_step=epoch)
            writer.add_scalar(tag='best_model/ACC2', scalar_value=best_stats['ACC2'], global_step=epoch)
            writer.add_scalar(tag='best_model/ACC3', scalar_value=best_stats['ACC3'], global_step=epoch)
            writer.add_scalar(tag='best_model/MIS', scalar_value=best_stats['MIS'], global_step=epoch)
            # Best checkpoint
            logging.info('Best Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 'epoch': epoch + 1},
                   p['best_checkpoint'])
        writer.add_scalar(tag='model/ACC1', scalar_value=best_stats['ACC1'], global_step=epoch)
        writer.add_scalar(tag='model/ACC2', scalar_value=best_stats['ACC2'], global_step=epoch)
        writer.add_scalar(tag='model/ACC3', scalar_value=best_stats['ACC3'], global_step=epoch)
        writer.add_scalar(tag='model/MIS', scalar_value=best_stats['MIS'], global_step=epoch)
        writer.flush()

        # Checkpoint
        logging.info('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 'epoch': epoch + 1},
                   p['last_checkpoint'])

    logger.info(eval_stats)
    logger.info(f"best_stats:\n{best_stats}")
    logger.info(f"train_time_total:\n{train_time_total}")
    logger.info(f"val_time_total:\n{val_time_total}")
    writer.add_hparams(hparam_dict={'batch_size': p['batch_size']},
                       metric_dict=eval_stats)


if __name__ == '__main__':
    main()
