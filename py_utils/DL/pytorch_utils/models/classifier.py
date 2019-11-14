
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from .mobilenetv2 import MobileNetV2
import os, math


class FPN(nn.Module):
    def __init__(self, backbone=101, num_class=2, pretrained=True):
        # Only resnet is supported in this version
        super(FPN, self).__init__()
        if backbone in [50, 101]:
            self.resnet = ResNet(backbone, num_class, pretrained)
        else:
            raise ValueError('Resnet{} is not supported yet.'.format(backbone))

        if backbone in [18, 34]:
            self.fc = nn.Linear(512, num_class)
        if backbone in [50, 101, 152]:
            self.fc = nn.Linear(512 * 4, num_class)

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.toplayer, 0, 0.01, False)
        normal_init(self.smooth1, 0, 0.01, False)
        normal_init(self.smooth2, 0, 0.01, False)
        normal_init(self.smooth3, 0, 0.01, False)
        normal_init(self.latlayer1, 0, 0.01, False)
        normal_init(self.latlayer2, 0, 0.01, False)
        normal_init(self.latlayer3, 0, 0.01, False)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def conv_base(self, x):
        # feed image data to base model to obtain base feature map
        # Bottom-up
        c2, c3, c4, c5 = self.resnet.conv_base(x)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)
        p6 = self.maxpool2d(p5)
        return p2, p3, p4, p5, p6

    def forward(self, x):
        p2, p3, p4, p5, p6 = self.conv_base(x)
        pass


class MobileNet(nn.Module):
    def __init__(self, version=1, num_class=2, pretrained=True):
        super(MobileNet, self).__init__()
        if version == 1:
            pass
        if version == 2:
            self.mobilenet = MobileNetV2(n_class=1000)
            if pretrained:
                home = os.path.expanduser("~")
                model_dir = os.path.join(home, '.pretrained')
                # Download pretrained model to ~/.pretrained
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_path = os.path.join(model_dir, 'mobilenetv2.pth.tar')
                if not os.path.exists(model_path):
                    import wget
                    wget.download(url='https://www.albany.edu/~yl149995/mobilenet_v2.pth.tar',
                                  out=model_path)
                state_dict = torch.load(model_path)  # add map_location='cpu' if no gpu
                self.mobilenet.load_state_dict(state_dict)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet.last_channel, num_class),
        )

    def conv_base(self, x):
        x = self.mobilenet.features(x)
        return x

    def forward(self, x):
        x = self.conv_base(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layers=18, num_class=2, pretrained=True):
        super(ResNet, self).__init__()
        if layers == 18:
            self.resnet = models.resnet18(pretrained=pretrained)
        elif layers == 34:
            self.resnet = models.resnet34(pretrained=pretrained)
        elif layers == 50:
            self.resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            self.resnet = models.resnet101(pretrained=pretrained)
        elif layers == 152:
            self.resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError('layers should be 18, 34, 50, 101.')
        self.num_class = num_class
        if layers in [18, 34]:
            self.fc = nn.Linear(512, num_class)
        if layers in [50, 101, 152]:
            self.fc = nn.Linear(512 * 4, num_class)

    def conv_base(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)
        return layer1, layer2, layer3, layer4

    def forward(self, x):
        layer1, layer2, layer3, layer4 = self.conv_base(x)
        x = self.resnet.avgpool(layer4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG(nn.Module):
    def __init__(self, layers=16, num_class=2, pretrained=True):
        super(VGG, self).__init__()
        if layers == 16:
            self.vgg = models.vgg16(pretrained=pretrained)
        elif layers == 19:
            self.vgg = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError('layers should be 16 or 19.')
        self.num_class = num_class
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )

    def conv_base(self, x):
        x = self.vgg.features(x)
        return x

    def forward(self, x):
        x = self.vgg.conv_base(x)
        x = self.vgg.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SqueezeNet(nn.Module):
    def __init__(self, version=1.0, num_class=2, pretrained=True):
        super(SqueezeNet, self).__init__()
        if version == 1.0:
            self.squeezenet = models.squeezenet1_0(pretrained=pretrained)
        elif version == 1.1:
            self.squeezenet = models.squeezenet1_1(pretrained=pretrained)
        else:
            raise ValueError('version should be 1.0 or 1.1.')

        self.num_class = num_class
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_class, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def conv_base(self, x):
        x = self.squeezenet.features(x)
        return x

    def forward(self, x):
        x = self.conv_base(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_class)


class DenseNet(nn.Module):
    def __init__(self, layers=121, num_class=2, pretrained=True):
        super(DenseNet, self).__init__()
        if layers == 121:
            self.densenet = models.densenet121(pretrained=pretrained)
        elif layers == 161:
            self.densenet = models.densenet161(pretrained=pretrained)
        elif layers == 169:
            self.densenet = models.densenet169(pretrained=pretrained)
        elif layers == 201:
            self.densenet = models.densenet201(pretrained=pretrained)
        else:
            raise ValueError('layers should be 121, 161, 169, 201.')

        self.num_class = num_class
        # Linear layer
        num_features = self.densenet.classifier.in_features
        self.classifier = nn.Linear(num_features, num_class)

    def conv_base(self, x):
        features = self.densenet.features(x)
        return features

    def forward(self, x):
        features = self.conv_base(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class InceptionNet(nn.Module):
    def __init__(self, version=3, num_class=2, pretrained=True):
        super(InceptionNet, self).__init__()
        if version == 3:
            self.squeezenet = models.inception_v3(pretrained=pretrained)
        else:
            raise ValueError('version should be 1.0 or 1.1.')
        self.num_class = num_class
        self.fc = nn.Linear(2048, num_class)

    def conv_base(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.squeezenet.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.squeezenet.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.squeezenet.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.squeezenet.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.squeezenet.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.squeezenet.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.squeezenet.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.squeezenet.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.squeezenet.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.squeezenet.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.squeezenet.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.squeezenet.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.squeezenet.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = []
        if self.squeezenet.training and self.squeezenet.aux_logits:
            aux = self.squeezenet.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.squeezenet.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.squeezenet.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.squeezenet.Mixed_7c(x)
        return x, aux

    def forward(self, x):
        x, aux = self.conv_base(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.squeezenet.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.squeezenet.training and self.squeezenet.aux_logits:
            return x, aux
        return x


class SPPNet(nn.Module):
    def __init__(self, backbone=101, num_class=2, pool_size=(1, 2, 6), pretrained=True):
        # Only resnet is supported in this version
        super(SPPNet, self).__init__()
        if backbone in [18, 34, 50, 101, 152]:
            self.resnet = ResNet(backbone, num_class, pretrained)
        else:
            raise ValueError('Resnet{} is not supported yet.'.format(backbone))

        if backbone in [18, 34]:
            self.c = 512
        if backbone in [50, 101, 152]:
            self.c = 2048

        self.spp = SpatialPyramidPool2D(out_side=pool_size)
        num_features = self.c * (pool_size[0] ** 2 + pool_size[1] ** 2 + pool_size[2] ** 2)
        self.classifier = nn.Linear(num_features, num_class)

    def forward(self, x):
        _, _, _, x = self.resnet.conv_base(x)
        x = self.spp(x)
        x = self.classifier(x)
        return x


class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        # batch_size, c, h, w = x.size()
        out = None
        for n in self.out_side:
            w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out


# pyramid pooling
class PPM(nn.Module):
    def __init__(self,
                 num_class=2,
                 fc_dim=128,
                 use_softmax=False,
                 pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        input_size = conv_out.size()
        ppm_out = [conv_out]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv_out),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        # x = nn.functional.interpolate(
        #     x, size=segSize, mode='bilinear', align_corners=False)
        # if self.use_softmax:  # is True during inference
        #     x = nn.functional.softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self,
                 num_class=150,
                 fc_dim=4096,
                 use_softmax=False,
                 pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
            "3x3 convolution with padding"
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=has_bias)

        def conv3x3_bn_relu(in_planes, out_planes, stride=1):
            return nn.Sequential(
                conv3x3(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )

        self.cbr_deepsup = nn.Sequential(
            conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1),
            nn.Dropout2d(0.1),
            nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        feat1 = conv_out[-1]
        input_size = feat1.size()
        ppm_out = [feat1]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(feat1),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        x = nn.functional.interpolate(
            x, size=segSize, mode='bilinear', align_corners=False)
        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
            return x
        else:
            # deep sup
            feat2 = conv_out[-2]
            x_ = self.cbr_deepsup(feat2)
            x_ = nn.functional.interpolate(
                x_, size=segSize, mode='bilinear', align_corners=False)
            return x, x_

