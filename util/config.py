import os
from datetime import datetime
import yaml
from easydict import EasyDict

from util.utils import mkdir_if_missing


def create_config(config_file, backbone=None, criterion=None):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v
    if backbone:
        cfg['backbone'] = backbone
    if criterion:
        cfg['criterion'] = criterion

    # Set paths for pretext task (These directories are needed in every stage)
    if not os.path.exists('./result'):
        os.mkdir('./result')
    dt = datetime.now().strftime('%m%d_%H%M')
    base_dir = f'./result/{dt}_' + cfg['db_name'] + f"_{cfg['backbone']}"+ f"_{cfg['criterion']}"
    if os.path.exists(base_dir):
        for i in range(10):
            base_dir = base_dir + f'_{i + 2}'
            if os.path.exists(base_dir):
                continue
            break
    os.mkdir(base_dir)

    cfg['base_dir'] = base_dir
    cfg['best_checkpoint'] = os.path.join(base_dir, 'best_checkpoint.pth.tar')
    cfg['last_checkpoint'] = os.path.join(base_dir, 'last_checkpoint.pth.tar')
    return cfg
