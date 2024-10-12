import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MyVggNet(nn.Module):
    def __init__(self):
        super(MyVggNet, self).__init__()
        # initial layer
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # SWP moudel
        self.swp = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1)
        )
        
        # fully connected layer
        self.fc1 = nn.Linear(128 * 64 * 32 * 8, 128)     # the maximum feature map size is 128x64x32x8
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)                 # the final output feature map size is 1
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.swp(x)
        # print('swp', x.shape)
        # feature map size is [BATCH_SIZE, 8, 64, 64, 32]
        x = x.view(x.size(0), -1)
        # print('view', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
# model = MyVggNet(num_classes=1)

# print(model)

class MyGoogleNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MyGoogleNet, self).__init__()

        base_model = models.video_resnet3d(pretrained=True)
        
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base_model.fc.in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        self.features[8] = nn.Identity()  

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# model = MyGoogleNet(num_classes=1)

# print(model)

class MyMobileNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MyMobileNet, self).__init__()
        # 使用预训练的MobileNetV2作为基础
        base_model = models.mobilenet_v2(pretrained=True)
        # self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.features = self._convert_to_3d(base_model.features)        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base_model.classifier[1].in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def _convert_to_3d(self, module):
        """
        2D --> 3D
        """
        if isinstance(module, nn.Conv2d):
            return nn.Conv3d(
                module.in_channels,
                module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                bias=module.bias is not None
            )
        elif isinstance(module, nn.BatchNorm2d):
            return nn.BatchNorm3d(module.num_features)
        elif isinstance(module, nn.ReLU):
            return nn.ReLU(inplace=module.inplace)
        elif isinstance(module, nn.MaxPool2d):
            return nn.MaxPool3d(kernel_size=module.kernel_size, stride=module.stride)
        elif isinstance(module, nn.Sequential):
            new_modules = []
            for m in module.modules():
                new_modules.append(self._convert_to_3d(m))
            return nn.Sequential(*new_modules)
        else:
            return module        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# model = MyMobileNet(num_classes=1)

# print(model)

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MyResNet(nn.Module):

    def __init__(self,
                 block=BasicBlock,
                 layers=[2, 2, 2, 2],
                 block_inplanes=[32, 64, 64, 32],
                 n_input_channels=2,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = MyResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = MyResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = MyResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = MyResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)

    return model