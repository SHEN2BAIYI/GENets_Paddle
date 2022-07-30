import os
import sys
import paddle
import paddle.nn as nn

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(BASE_DIR)

__all__ = [
    'GENet_light',
    'GENet_normal',
    'GENet_large',
]

GENet_type_config = {
    'light': {
        'stages_depth': '[1, 1, 3, 7, 2, 1, 1]',
        'stages_channel': '[13, 48, 48, 384, 560, 256, 1920]',
        'stages_stride': '[2, 2, 2, 2, 2, 1, 1]',
        'stages_neck_ratio': '[1, 1, 1, 0.25, 3, 3, 1]',
    },

    'normal': {
        'stages_depth': '[1, 1, 2, 6, 4, 1, 1]',
        'stages_channel': '[32, 128, 192, 640, 640, 640, 2560]',
        'stages_stride': '[2, 2, 2, 2, 2, 1, 1]',
        'stages_neck_ratio': '[1, 1, 1, 0.25, 3, 3, 1]',
    },

    'large': {
        'stages_depth': '[1, 1, 2, 6, 5, 4, 1]',
        'stages_channel': '[32, 128, 192, 640, 640, 640, 2560]',
        'stages_stride': '[2, 2, 2, 2, 2, 1, 1]',
        'stages_neck_ratio': '[1, 1, 1, 0.25, 3, 3, 1]',
    }

}


class ConvBnActBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, has_bn=True, has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act

        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                              bias_attr=False)

        if self.has_bn:
            self.bn = nn.BatchNorm2D(out_channels)
        if self.has_act:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.act(x)
        return x


class XBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, groups=1, neck_ratio=1, downsample=False):
        super(XBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.downsample_layer = ConvBnActBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                                   groups=1, has_bn=True, has_act=False)

        self.conv1 = ConvBnActBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=1,
                                    has_bn=True, has_act=True)
        self.conv2 = ConvBnActBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1,
                                    has_bn=True, has_act=False)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.downsample_layer(inputs)

        x += inputs
        x = self.relu(x)
        return x


class BBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, groups=1, neck_ratio=1, downsample=False):
        super(BBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.downsample_layer = ConvBnActBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                                   has_bn=True, has_act=False)

        self.conv1 = ConvBnActBlock(in_channels, int(out_channels * neck_ratio), kernel_size=1, stride=1, padding=0,
                                    groups=1, has_bn=True, has_act=True)
        self.conv2 = ConvBnActBlock(int(out_channels * neck_ratio), int(out_channels * neck_ratio), kernel_size=3,
                                    stride=stride, padding=1, groups=groups, has_bn=True, has_act=True)
        self.conv3 = ConvBnActBlock(int(out_channels * neck_ratio), out_channels, kernel_size=1, stride=1, padding=0,
                                    groups=1, has_bn=True, has_act=False)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample:
            inputs = self.downsample_layer(inputs)

        x += inputs
        x = self.relu(x)

        return x


class DBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, groups=1, neck_ratio=1, downsample=False):
        super(DBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.downsample_layer = ConvBnActBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                                   has_bn=True, has_act=False)

        self.conv1 = ConvBnActBlock(in_channels, int(out_channels * neck_ratio), kernel_size=1, stride=1, padding=0,
                                    groups=1, has_bn=True, has_act=True)
        self.conv2 = ConvBnActBlock(int(out_channels * neck_ratio), int(out_channels * neck_ratio), kernel_size=3,
                                    stride=stride, padding=1, groups=int(out_channels * neck_ratio), has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(int(out_channels * neck_ratio), out_channels, kernel_size=1, stride=1, padding=0,
                                    groups=1, has_bn=True, has_act=False)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample:
            inputs = self.downsample_layer(inputs)

        x += inputs
        x = self.relu(x)
        return x


class GENet(nn.Layer):
    def __init__(self, ge_type, in_channels=3, num_classes=1000):
        super(GENet, self).__init__()

        ge_config = GENet_type_config[ge_type]
        self.stages_neck_ratio = eval(ge_config['stages_neck_ratio'])
        self.stages_stride = eval(ge_config['stages_stride'])
        self.stages_depth = eval(ge_config['stages_depth'])
        self.stages_channel = eval(ge_config['stages_channel'])

        # self.stages_neck_ratio = [1, 1, 1, 0.25, 3, 3, 1]
        # self.stages_stride = [2, 2, 2, 2, 2, 1, 1]
        # self.stages_depth = [1, 1, 3, 7, 2, 1, 1]
        # self.stages_channel = [13, 48, 48, 384, 560, 256, 1920]

        # Stem
        self.begin = ConvBnActBlock(in_channels, self.stages_channel[0], kernel_size=3, stride=self.stages_stride[0],
                                    padding=1, groups=1, has_bn=True, has_act=True)
        self.end = ConvBnActBlock(self.stages_channel[-2], self.stages_channel[-1], kernel_size=1,
                                  stride=self.stages_stride[-1], padding=0, groups=1, has_bn=True, has_act=True)

        # Body
        self.stage1 = self.make_stage(self.stages_channel[0], self.stages_channel[1], stride=self.stages_stride[1],
                                      neck_ratio=self.stages_neck_ratio[1], block_num=self.stages_depth[1],
                                      block_type=XBlock)
        self.stage2 = self.make_stage(self.stages_channel[1], self.stages_channel[2], stride=self.stages_stride[2],
                                      neck_ratio=self.stages_neck_ratio[2], block_num=self.stages_depth[2],
                                      block_type=XBlock)
        self.stage3 = self.make_stage(self.stages_channel[2], self.stages_channel[3], stride=self.stages_stride[3],
                                      neck_ratio=self.stages_neck_ratio[3], block_num=self.stages_depth[3],
                                      block_type=BBlock)
        self.stage4 = self.make_stage(self.stages_channel[3], self.stages_channel[4], stride=self.stages_stride[4],
                                      neck_ratio=self.stages_neck_ratio[4], block_num=self.stages_depth[4],
                                      block_type=DBlock)
        self.stage5 = self.make_stage(self.stages_channel[4], self.stages_channel[5], stride=self.stages_stride[5],
                                      neck_ratio=self.stages_neck_ratio[5], block_num=self.stages_depth[5],
                                      block_type=DBlock)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(self.stages_channel[-1], num_classes)

        # Init Parameters
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight_attr = nn.initializer.KaimingNormal()
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                m.weight_attr = nn.initializer.Constant(value=1.)
                m.bias_attr = nn.initializer.Constant(value=0.)

    def make_stage(self, in_channels, out_channels, stride, neck_ratio, block_num, block_type):
        layers = []
        for block_index in range(block_num):
            downsample = True if block_index == 0 and (stride != 1 or in_channels != out_channels) else False

            in_channels = out_channels if block_index > 0 else in_channels
            stride = 1 if block_index > 0 else stride
            layers.append(
                block_type(in_channels, out_channels, stride=stride, neck_ratio=neck_ratio, downsample=downsample)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.begin(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.end(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x


def _genet(ge_type, pretrained, **kwargs):
    model = GENet(ge_type, **kwargs)
    if pretrained:
        pass

    return model


def GENet_light(pretrained=False, **kwargs):
    return _genet('light', pretrained, **kwargs)


def GENet_normal(pretrained=False, **kwargs):
    return _genet('normal', pretrained, **kwargs)


def GENet_large(pretrained=False, **kwargs):
    return _genet('large', pretrained, **kwargs)












