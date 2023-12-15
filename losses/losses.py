import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassSpecificCE(nn.Module):
    def __init__(self, lambda_weight=1):
        super(ClassSpecificCE, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, inputs, targets):
        pred_1 = inputs['pred_1']
        pred_2 = inputs['pred_2']
        pred_3 = inputs['pred_3']
        ind_positive_sample = inputs['ind_positive_sample']
        with torch.no_grad():
            n_positive_sample = int(torch.sum(ind_positive_sample))
            ACC1 = int((pred_1.argmax(-1) == targets).sum()) / targets.shape[0]
            ACC2 = int((pred_2.argmax(-1) == targets).sum()) / targets.shape[0]
            ACC3 = int((pred_3.argmax(-1) == targets).sum()) / targets.shape[0]
        loss_discrimination = F.cross_entropy(pred_1, targets)  # included 'softmax'
        if n_positive_sample != 0:
            loss_interpretation = F.cross_entropy(pred_2[ind_positive_sample], targets[ind_positive_sample])
        else:
            loss_interpretation = torch.tensor(0.0).cuda()
        loss_total = loss_discrimination + loss_interpretation * self.lambda_weight
        return {"loss_discrimination": loss_discrimination, "loss_interpretation": loss_interpretation, "loss_total": loss_total,
                "n_positive_sample": n_positive_sample, 'ACC1': ACC1, 'ACC2': ACC2,'ACC3': ACC3}


class StandardCE(nn.Module):
    def __init__(self):
        super(StandardCE, self).__init__()

    def forward(self, inputs, targets):
        pred_1 = inputs['pred_1']
        pred_2 = inputs['pred_2']
        pred_3 = inputs['pred_3']
        loss_discrimination = F.cross_entropy(pred_1, targets)  # included 'softmax'
        loss_total = loss_discrimination
        with torch.no_grad():
            ACC1 = int((pred_1.argmax(-1) == targets).sum()) / targets.shape[0]
            ACC2 = int((pred_2.argmax(-1) == targets).sum()) / targets.shape[0]
            ACC3 = int((pred_3.argmax(-1) == targets).sum()) / targets.shape[0]
        return {"loss_discrimination": loss_discrimination, "loss_interpretation": 0, "loss_total": loss_total,
                "n_positive_sample": pred_1.shape[0], 'ACC1': ACC1, 'ACC2': ACC2,'ACC3': ACC3}


