import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):

    def __init__(self, depth, num_classes=100, block_name='BasicBlock'):
        super(PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        p = x.view(x.size(0), -1)
        p = self.fc(p)

        return p, x




def preresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return PreResNet(**kwargs)


def get_parameter_number(net):    
        total_num = sum(p.numel() for p in net.parameters())    
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)    
        # return {'Total': total_num, 'Trainable': trainable_num}
        print('Total: ', total_num, '    Trainable: ', trainable_num)


def get_mask_size(net):
    for module in net.modules():
        if isinstance(module, nn.Conv2d) and module.weight.data.shape[1]>=16 and module.weight.data.shape[2] == 3:
            print(module.weight.data.shape)



class Generator(nn.Module):
    def __init__(self, in_features, out_features, total_class, hidden_dim, num_units, units_x, units_y, seg):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_class = total_class
        self.hidden_dim = hidden_dim
        self.num_units = num_units
        self.units_x = units_x
        self.units_y = units_y
        self.seg = seg  # [1]*4 + [2]*4 + [4]*4 + [8]*4  for Resnet-18

        self.fc_1 = nn.Linear(self.total_class, self.hidden_dim)
        self.fc_2 = nn.Linear(self.in_features, self.out_features)
        # self.old_model = None
        self.bn_1 = nn.BatchNorm1d(self.hidden_dim)

        # self.units = torch.rand(self.num_units, self.units_x, self.units_y)  # 30*16*9


        assert sum(self.seg) * self.num_units == self.out_features


    def forward(self, emb, label, units):
        x = F.relu(self.bn_1(self.fc_1(label)))
        x = torch.cat([emb, x], -1)
        results = torch.sigmoid(self.fc_2(x)) # batch_size * (sum(self.seg) * self.num_units)   20 * 1800

        indexes = [] # batch_size * (num_units , 2*num_units, 4*num_units, 8*num_units) 20*30 20*60 20*120 20*240
        l = np.cumsum([0] + self.seg) * self.num_units
        for s, e in zip(l[:-1], l[1:]):
            indexes.append(results[:, s:e])
        
        masks = []
        for i in range(len(self.seg)):
            for n in range(int(indexes[i].shape[1] / self.num_units)):
                if n == 0:
                    # print("==============================")
                    # print("emb:",indexes[i][:,:(n+1)*self.num_units].shape)
                    # print("x:",units.shape)
                    # print("==============================")
                    masks.append(torch.sum(indexes[i][:,:(n+1)*self.num_units].unsqueeze(-1).unsqueeze(-1) * units, dim=1) / self.num_units)
                else:
                    masks[i] = torch.cat([masks[i], torch.sum(indexes[i][:,n*self.num_units:(n+1)*self.num_units].unsqueeze(-1).unsqueeze(-1) * units, dim=1) / self.num_units], 1)
            
            assert masks[i].shape[1] ==  self.units_x * int(indexes[i].shape[1] / self.num_units)   # batch_size * 64/128/256/512 * 9

        return masks





# if __name__ == "__main__":
#     net = ResNet18()
#     fisher = {}
#     for n,p in net.named_parameters():
#         print(n)
#         fisher[n] = 0 * p.data
#         print(fisher[n].size())
#         print("------------------------")
    
#     get_parameter_number(net)
    
    


