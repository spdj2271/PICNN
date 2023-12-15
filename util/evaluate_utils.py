from multiprocessing import Pool
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False):
    model.eval()

    features = []
    predictions_1 = []
    predictions_2 = []
    predictions_3 = []
    targets = []
    
    for batch in dataloader:
        images_bc = batch['image'].cuda(non_blocking=True)
        targets_bc = batch['target'].cuda(non_blocking=True)
        out = model(images_bc, targets=targets_bc, forward_pass='all')
        features.append(torch.sum(out['features'],(-1,-2)))
        predictions_1.append(torch.argmax(out['pred_1'], dim=1))
        predictions_2.append(torch.argmax(out['pred_2'], dim=1))
        predictions_3.append(torch.argmax(out['pred_3'], dim=1))
        targets.append(batch['target'])

    features = torch.cat(features).cpu().numpy()
    predictions_1 = torch.cat(predictions_1, dim=0).cpu().numpy()
    predictions_2 = torch.cat(predictions_2, dim=0).cpu().numpy()
    predictions_3 = torch.cat(predictions_3, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).cpu().numpy()
    
    out = {"features": features, 'pred_1': predictions_1, 'pred_2': predictions_2, 'pred_3': predictions_3,
           'targets': targets }
    return out


def compute_MIS(features, targets, c):
    print(f'class {c}', end='; ')
    return mutual_info_classif(features, targets == c, random_state=0, discrete_features=False)


@torch.no_grad()
def evaluate_ACC_MIS(predictions, correlation,epoch):
    targets = predictions['targets']
    pred_1 = predictions['pred_1']
    pred_2 = predictions['pred_2']
    pred_3 = predictions['pred_3']
    features = predictions['features']
    ACC1 = int((pred_1 == targets).sum()) / float(targets.shape[0])
    ACC2 = int((pred_2 == targets).sum()) / float(targets.shape[0])
    ACC3 = int((pred_3 == targets).sum()) / float(targets.shape[0])

    M_info = []
    M_info_balanced = []
    print('computing MIS ...')
    labels = np.unique(targets)
    pool = Pool()
    results = []
    for c in labels:
        result = pool.apply_async(compute_MIS, (features, targets, c))
        results.append(result)
    pool.close()
    pool.join()
    for result in results:
        M_info.append(result.get())
    print()
    MutaulInfo = np.stack(M_info)
    MIS = np.mean(np.max(MutaulInfo, axis=0))

    return {'ACC1': ACC1, "ACC2": ACC2, 'ACC3': ACC3, "MIS": MIS}