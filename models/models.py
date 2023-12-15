import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoadModel(nn.Module):
    def __init__(self, p):
        super(LoadModel, self).__init__()
        self.n_classes = p['num_classes']
        self.n_filters = p['model_kwargs']['features_dim']

        if p.backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.weights = ResNet18_Weights.DEFAULT
            base_model = resnet18(weights=self.weights)
            modules = list(base_model.children())
            new_conv_layer = nn.Conv2d(512, self.n_filters, 3, padding=1)
            modules.insert(8, new_conv_layer)
            modules[-1] = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=self.n_filters, out_features=self.n_classes))
            self.backbone = nn.Sequential(*modules[:-2])
            self.classifier = nn.Sequential(*modules[-2:])


        else:
            raise Exception(f"unknown base model {p.backbone}")
        self.correlation = nn.Parameter(F.softmax(torch.rand(size=(self.n_classes, self.n_filters)), dim=0))

    def forward(self, inputs, targets=None, forward_pass='default'):
        features = self.backbone(inputs)
        pred = self.classifier(features)
        pred_1 = pred # prediction of discrimination pathway (using all filters)

        # sample class ID using reparameter trick
        pred_softmax = torch.softmax(pred, dim=-1)
        with torch.no_grad():
            sample_cat = torch.multinomial(pred_softmax, 1, replacement=False).flatten().cuda()
            ind_positive_sample = sample_cat == targets  # mark wrong sample results
            sample_cat_oh = F.one_hot(sample_cat, num_classes=pred.shape[1]).float().cuda()
            epsilon = torch.where(sample_cat_oh != 0, 1 - pred_softmax, -pred_softmax).detach()
        sample_cat_oh = pred_softmax + epsilon

        # sample filter using reparameter trick
        correlation_softmax = F.softmax(self.correlation, dim=0)
        correlation_samples = sample_cat_oh @ correlation_softmax
        with torch.no_grad():
            ind_sample = torch.bernoulli(correlation_samples).bool()
            epsilon = torch.where(ind_sample, 1 - correlation_samples, -correlation_samples)
        binary_mask = correlation_samples + epsilon
        feature_mask = features * binary_mask[..., None, None]  # binary
        pred_2 = self.classifier(feature_mask) # prediction of Interpretation pathway (using a cluster of class-specific filters)
        with torch.no_grad(): 
            correlation_samples = correlation_softmax[targets]
            binary_mask = torch.bernoulli(correlation_samples).bool()
            feature_mask_self = features * ~binary_mask[..., None, None]
            pred_3 = self.classifier(feature_mask_self) # prediction of Interpretation pathway (using complementary clusters of class-specific filters)
        out = {"features": features, 'pred_1': pred_1, 'pred_2': pred_2, 'pred_3': pred_3,
                    'ind_positive_sample': ind_positive_sample}
        return out

