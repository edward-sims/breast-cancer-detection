import torch
import torch.nn as nn
import pretrainedmodels
import ssl

from utils.constants import *


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=8, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class se_resnext101_32x4d(nn.Module):
    def __init__(self):
        super(se_resnext101_32x4d, self).__init__()
        
        # Workaround for SSL errors on calling pretrained models
        # https://github.com/pytorch/pytorch/issues/33288#issuecomment-954160699
        ssl._create_default_https_context = ssl._create_unverified_context
        
        self.model_ft = nn.Sequential(
            *list(pretrainedmodels.__dict__[MODEL_NAME](num_classes=1000, pretrained="imagenet").children())[
                :-2
            ]
        )   
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(8, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        #fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output