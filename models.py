import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from functions import ReverseLayerF

class Extractor(nn.Module):
    def __init__(self, lmd1, lmd2):
        super(Extractor, self).__init__()
        inplane = 1
        outplane = 32
        midplanes = int(32 * lmd2)
        mid = int(4 * outplane * lmd1)
        self.layer1 = nn.Sequential(
            nn.Conv3d(inplane, mid, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, midplanes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(midplanes, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, outplane, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outplane, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.max1 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=0, stride=2))

        inplane = 32
        outplane = 128
        midplanes = int(inplane * lmd2)
        mid = int(4 * outplane * lmd1)
        self.layer2 = nn.Sequential(
            nn.Conv3d(inplane, mid, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, midplanes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(midplanes, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, outplane, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outplane, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.max2 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1, stride=2))

        inplane = 128
        outplane = 256
        midplanes = int(inplane * lmd2)
        mid = int(4 * outplane * lmd1)
        self.layer3 = nn.Sequential(
            nn.Conv3d(inplane, mid, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, midplanes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(midplanes, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, outplane, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outplane, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.max3 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1, stride=2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.max1(x)

        x = self.layer2(x)
        x = self.max2(x)

        x = self.layer3(x)
        x = self.max3(x)

        return x

class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class MASS(nn.Module):

    def __init__(self, num_classes=10, num_domains = 14):
        super(MASS, self).__init__()
        self.num_domains = num_domains
        '''Net of CFE'''
        self.sharedNet = Extractor(1, 1)
        '''Net of DSFE (Feature Extractor) in DSR'''
        self.addnetlist = nn.ModuleList([ADDneck(256, 64) for i in range(self.num_domains)])
        '''Net of DSC (Classifier) in DSR'''
        self.fcnetlist = nn.ModuleList([nn.Linear(64, num_classes) for i in range(self.num_domains)])
        '''Net of (Domain Classifier) in DSR'''
        self.domain_fcnetlist = nn.ModuleList([nn.Linear(64, 2) for i in range(self.num_domains)])
        '''Net of (Classifier) in DGR'''
        self.fcnet_all = nn.Linear(256, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, data_src, data_tgt=0, label_src=0, mark=0, alpha = 1):
        # mmd_loss = 0
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt_son = []
            data_pre_son = []
            data_domain_pre_son = []
            for i in range(self.num_domains):
                data_tgt_son1 = self.addnetlist[i](data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                reverse_feature = ReverseLayerF.apply(data_tgt_son1, alpha)
                pred_tgt_son1 = self.fcnetlist[i](data_tgt_son1)
                domn_tgt_son1 = self.domain_fcnetlist[i](reverse_feature)
                data_tgt_son.append(data_tgt_son1)
                data_pre_son.append(pred_tgt_son1)
                data_domain_pre_son.append(domn_tgt_son1)

            trainlist = list(range(self.num_domains))
            for i in range(self.num_domains):
                if mark == i:
                    fea_src = self.avgpool(data_src)
                    fea_src = fea_src.view(fea_src.size(0), -1)
                    pred_src_whole = self.fcnet_all(fea_src)     #classifier in DGR

                    data_src = self.addnetlist[mark](data_src)
                    data_src = self.avgpool(data_src)
                    data_src = data_src.view(data_src.size(0), -1)

                    l1_loss = 0
                    trainlist.pop(mark)
                    for j in range(self.num_domains - 1):
                        l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_pre_son[mark], dim=1)
                                                        - torch.nn.functional.softmax(data_pre_son[trainlist[j]], dim=1)))
                    pred_src = self.fcnetlist[i](data_src)    #classifier in DSR

                    reverse_feature = ReverseLayerF.apply(data_src, alpha)
                    domain_pred_src = self.domain_fcnetlist[i](reverse_feature)    #domain classifier in DSR
                    src_domian_loss = F.nll_loss(F.log_softmax(domain_pred_src, dim=1), torch.zeros(len(domain_pred_src)).long().cuda())
                    tgt_domian_loss = F.nll_loss(F.log_softmax(data_domain_pre_son[mark], dim=1), torch.ones(len(data_domain_pre_son[mark])).long().cuda())

                    cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                    cls_loss_whole = F.nll_loss(F.log_softmax(pred_src_whole, dim=1), label_src)

                    return cls_loss_whole, cls_loss, src_domian_loss, tgt_domian_loss, l1_loss / 2

        else:
            data = self.sharedNet(data_src)
            pred_all = []
            fea_src = self.avgpool(data)
            fea_src = fea_src.view(fea_src.size(0), -1)
            pred_src_whole = self.fcnet_all(fea_src)  # classifier in DGR
            pred_all.append(pred_src_whole)
            for i in range(self.num_domains):
                fea_son1 = self.addnetlist[i](data)
                fea_son1 = self.avgpool(fea_son1)
                fea_son1 = fea_son1.view(fea_son1.size(0), -1)
                pred_tgt_son1 = self.fcnetlist[i](fea_son1)    #domain classifier in DSR
                pred_all.append(pred_tgt_son1)

            return pred_all

# def modeltest():
#     net = Extractor(1, 1)
#     x1 = torch.randn(32, 1, 48, 10, 10)
#     x2 = torch.randn(32, 1, 48, 10, 10)
#     y = torch.empty(32, dtype=torch.long).random_(3)
#     gesnet = MASS(10,14)
#     gesnet = gesnet.cuda()
#     print(gesnet)
#     output = gesnet(x1.cuda(), x2.cuda(), y.cuda(),mark=0)
#     print(output)
# modeltest()