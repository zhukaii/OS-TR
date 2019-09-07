import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision import models
import math

# a = torch.randn(1, 3, 256, 256)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('utils/model/resnet50-19c8e357.pth'))
    return model


def deep_orientation_gen(ins):
    # Up and Down
    ins_U = ins
    ins_D = ins
    # Left and Right
    ins_L = ins
    ins_R = ins
    # Left-Up and Right-Down
    ins_LU = ins
    ins_RD = ins
    # Right-Up and Left-Down
    ins_RU = ins
    ins_LD = ins

    batch_size_tensor, c, w, h = ins.size()

    # Up
    U_ones = torch.ones([1, w], dtype=torch.int)
    U_ones = U_ones.unsqueeze(-1)
    U_range = torch.arange(w, dtype=torch.int).unsqueeze(0)
    U_range = U_range.unsqueeze(1)
    U_channel = torch.matmul(U_ones, U_range)
    U_channel = U_channel.unsqueeze(-1)
    U_channel = U_channel.permute(0, 3, 1, 2)
    U_channel = U_channel.float() / (w - 1)
    U_channel = U_channel * 2 - 1
    U_channel = U_channel.repeat(batch_size_tensor, 1, 1, 1)
    U_channel = U_channel.cuda()
    ins_U_new = torch.cat((ins_U, U_channel), 1)
    # ins_U_new = ins_U * U_channel

    # Down
    D_ones = torch.ones([1, w], dtype=torch.int)
    D_ones = D_ones.unsqueeze(-1)
    D_range = torch.arange(w - 1, -1, -1, dtype=torch.int).unsqueeze(0)
    D_range = D_range.unsqueeze(1)
    D_channel = torch.matmul(D_ones, D_range)
    D_channel = D_channel.unsqueeze(-1)
    D_channel = D_channel.permute(0, 3, 1, 2)
    D_channel = D_channel.float() / (w - 1)
    D_channel = D_channel * 2 - 1
    D_channel = D_channel.repeat(batch_size_tensor, 1, 1, 1)
    D_channel = D_channel.cuda()
    ins_D_new = torch.cat((ins_D, D_channel), 1)
    # ins_D_new = ins_D * D_channel

    # Left
    L_ones = torch.ones([1, h], dtype=torch.int)
    L_ones = L_ones.unsqueeze(1)
    L_range = torch.arange(h, dtype=torch.int).unsqueeze(0)
    L_range = L_range.unsqueeze(-1)
    L_channel = torch.matmul(L_range, L_ones)
    L_channel = L_channel.unsqueeze(-1)
    L_channel = L_channel.permute(0, 3, 1, 2)
    L_channel = L_channel.float() / (h - 1)
    L_channel = L_channel * 2 - 1
    L_channel = L_channel.repeat(batch_size_tensor, 1, 1, 1)
    L_channel = L_channel.cuda()
    ins_L_new = torch.cat((ins_L, L_channel), 1)
    # ins_L_new = ins_L * L_channel

    # Right
    R_ones = torch.ones([1, h], dtype=torch.int)
    R_ones = R_ones.unsqueeze(1)
    R_range = torch.arange(h - 1, -1, -1, dtype=torch.int).unsqueeze(0)
    R_range = R_range.unsqueeze(-1)
    R_channel = torch.matmul(R_range, R_ones)
    R_channel = R_channel.unsqueeze(-1)
    R_channel = R_channel.permute(0, 3, 1, 2)
    R_channel = R_channel.float() / (h - 1)
    R_channel = R_channel * 2 - 1
    R_channel = R_channel.repeat(batch_size_tensor, 1, 1, 1)
    R_channel = R_channel.cuda()
    ins_R_new = torch.cat((ins_R, R_channel), 1)
    # ins_R_new = ins_R * R_channel

    # Left and Up
    LU_ones_1 = torch.ones([w, h], dtype=torch.int)
    LU_ones_1 = torch.triu(LU_ones_1)
    LU_ones_2 = torch.ones([w, h], dtype=torch.int)
    LU_change = torch.arange(h - 1, -1, -1, dtype=torch.int)
    LU_ones_2[w - 1, :] = LU_change
    LU_channel = torch.matmul(LU_ones_1, LU_ones_2)
    LU_channel = LU_channel.unsqueeze(0).unsqueeze(-1)
    LU_channel = LU_channel.permute(0, 3, 1, 2)
    LU_channel = LU_channel.float() / (h - 1)
    LU_channel = LU_channel * 2 - 1
    LU_channel = LU_channel.repeat(batch_size_tensor, 1, 1, 1)
    LU_channel = LU_channel.cuda()
    ins_LU_new = torch.cat((ins_LU, LU_channel), 1)
    # ins_LU_new = ins_LU * LU_channel

    # Right and Down
    RD_ones_1 = torch.ones([w, h], dtype=torch.int)
    RD_ones_1 = torch.triu(RD_ones_1)
    RD_ones_1 = torch.t(RD_ones_1)
    RD_ones_2 = torch.ones([w, h], dtype=torch.int)
    RD_change = torch.arange(h, dtype=torch.int)
    RD_ones_2[0, :] = RD_change
    RD_channel = torch.matmul(RD_ones_1, RD_ones_2)
    RD_channel = RD_channel.unsqueeze(0).unsqueeze(-1)
    RD_channel = RD_channel.permute(0, 3, 1, 2)
    RD_channel = RD_channel.float() / (h - 1)
    RD_channel = RD_channel * 2 - 1
    RD_channel = RD_channel.repeat(batch_size_tensor, 1, 1, 1)
    RD_channel = RD_channel.cuda()
    ins_RD_new = torch.cat((ins_RD, RD_channel), 1)
    # ins_RD_new = ins_RD * RD_channel

    # Right and Up
    RU_ones_1 = torch.ones([w, h], dtype=torch.int)
    RU_ones_1 = torch.triu(RU_ones_1)
    RU_ones_2 = torch.ones([w, h], dtype=torch.int)
    RU_change = torch.arange(h, dtype=torch.int)
    RU_ones_2[w - 1, :] = RU_change
    RU_channel = torch.matmul(RU_ones_1, RU_ones_2)
    RU_channel = RU_channel.unsqueeze(0).unsqueeze(-1)
    RU_channel = RU_channel.permute(0, 3, 1, 2)
    RU_channel = RU_channel.float() / (h - 1)
    RU_channel = RU_channel * 2 - 1
    RU_channel = RU_channel.repeat(batch_size_tensor, 1, 1, 1)
    RU_channel = RU_channel.cuda()
    ins_RU_new = torch.cat((ins_RU, RU_channel), 1)
    # ins_RU_new = ins_RU * RU_channel

    # Left and Down
    LD_ones_1 = torch.ones([w, h], dtype=torch.int)
    LD_ones_1 = torch.triu(LD_ones_1)
    LD_ones_1 = torch.t(LD_ones_1)
    LD_ones_2 = torch.ones([w, h], dtype=torch.int)
    LD_change = torch.arange(h - 1, -1, -1, dtype=torch.int)
    LD_ones_2[0, :] = LD_change
    LD_channel = torch.matmul(LD_ones_1, LD_ones_2)
    LD_channel = LD_channel.unsqueeze(0).unsqueeze(-1)
    LD_channel = LD_channel.permute(0, 3, 1, 2)
    LD_channel = LD_channel.float() / (h - 1)
    LD_channel = LD_channel * 2 - 1
    LD_channel = LD_channel.repeat(batch_size_tensor, 1, 1, 1)
    LD_channel = LD_channel.cuda()
    ins_LD_new = torch.cat((ins_LD, LD_channel), 1)
    # ins_LD_new = ins_LD * LD_channel

    return ins_U_new, ins_D_new, ins_L_new, ins_R_new, ins_LU_new, ins_RD_new, ins_RU_new, ins_LD_new


class Deep_Orientation(nn.Module):
    def __init__(self, input_channel, output_channel, mid_channel):
        super(Deep_Orientation, self).__init__()

        self.transition_1 = nn.Conv2d(input_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.transition_1_bn = nn.BatchNorm2d(input_channel)
        self.transition_2_U = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.transition_2_U_bn = nn.BatchNorm2d(1 + mid_channel)
        self.transition_2_D = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.transition_2_D_bn = nn.BatchNorm2d(1 + mid_channel)
        self.transition_2_L = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.transition_2_L_bn = nn.BatchNorm2d(1 + mid_channel)
        self.transition_2_R = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                        bias=False, dilation=1)
        self.transition_2_R_bn = nn.BatchNorm2d(1 + mid_channel)
        self.transition_2_LU = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_LU_bn = nn.BatchNorm2d(1 + mid_channel)
        self.transition_2_RD = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_RD_bn = nn.BatchNorm2d(1 + mid_channel)
        self.transition_2_RU = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_RU_bn = nn.BatchNorm2d(1 + mid_channel)
        self.transition_2_LD = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_LD_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_3 = nn.Conv2d(mid_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.transition_3_bn = nn.BatchNorm2d(mid_channel)

        self.scale = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(True),
            nn.Linear(8, 8),
            # nn.Sigmoid(),
            nn.Softmax(dim=1),
        )

    def forward(self, x, stage=None):
        x = F.relu(self.transition_1(self.transition_1_bn(x)))

        ins_U_new, ins_D_new, ins_L_new, ins_R_new, ins_LU_new, ins_RD_new, ins_RU_new, ins_LD_new = deep_orientation_gen(x)

        ins_U_new = F.relu(self.transition_2_U(self.transition_2_U_bn(ins_U_new)))
        ins_D_new = F.relu(self.transition_2_D(self.transition_2_D_bn(ins_D_new)))
        ins_L_new = F.relu(self.transition_2_L(self.transition_2_L_bn(ins_L_new)))
        ins_R_new = F.relu(self.transition_2_R(self.transition_2_R_bn(ins_R_new)))
        ins_LU_new = F.relu(self.transition_2_LU(self.transition_2_LU_bn(ins_LU_new)))
        ins_RD_new = F.relu(self.transition_2_RD(self.transition_2_RD_bn(ins_RD_new)))
        ins_RU_new = F.relu(self.transition_2_RU(self.transition_2_RU_bn(ins_RU_new)))
        ins_LD_new = F.relu(self.transition_2_LD(self.transition_2_LD_bn(ins_LD_new)))

        batch = ins_U_new.shape[0]
        scale_U, _ = ins_U_new.reshape((batch, -1)).max(1)
        scale_D, _ = ins_U_new.reshape((batch, -1)).max(1)
        scale_L, _ = ins_U_new.reshape((batch, -1)).max(1)
        scale_R, _ = ins_U_new.reshape((batch, -1)).max(1)
        scale_LU, _ = ins_U_new.reshape((batch, -1)).max(1)
        scale_RD, _ = ins_U_new.reshape((batch, -1)).max(1)
        scale_RU, _ = ins_U_new.reshape((batch, -1)).max(1)
        scale_LD, _ = ins_U_new.reshape((batch, -1)).max(1)

        scale_U = scale_U.unsqueeze(1)
        scale_D = scale_D.unsqueeze(1)
        scale_L = scale_L.unsqueeze(1)
        scale_R = scale_R.unsqueeze(1)
        scale_LU = scale_LU.unsqueeze(1)
        scale_RD = scale_RD.unsqueeze(1)
        scale_RU = scale_RU.unsqueeze(1)
        scale_LD = scale_LD.unsqueeze(1)

        scale = torch.cat((scale_U, scale_D, scale_L, scale_R, scale_LU, scale_RD, scale_RU, scale_LD), 1)
        scale = self.scale(scale)

        ins_U_new = scale[:, 0:1].unsqueeze(2).unsqueeze(3) * ins_U_new
        ins_D_new = scale[:, 1:2].unsqueeze(2).unsqueeze(3) * ins_D_new

        ins_L_new = scale[:, 2:3].unsqueeze(2).unsqueeze(3) * ins_L_new
        ins_R_new = scale[:, 3:4].unsqueeze(2).unsqueeze(3) * ins_R_new

        ins_LU_new = scale[:, 4:5].unsqueeze(2).unsqueeze(3) * ins_LU_new
        ins_RD_new = scale[:, 5:6].unsqueeze(2).unsqueeze(3) * ins_RD_new

        ins_RU_new = scale[:, 6:7].unsqueeze(2).unsqueeze(3) * ins_RU_new
        ins_LD_new = scale[:, 7:8].unsqueeze(2).unsqueeze(3) * ins_LD_new

        x = torch.cat((ins_U_new, ins_D_new, ins_L_new, ins_R_new, ins_LU_new, ins_RD_new, ins_RU_new, ins_LD_new), 1)
        out = F.relu(self.transition_3(self.transition_3_bn(x)))

        return out


class encode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.resnet_layer1 = nn.Sequential(*list(model.children())[0:5])
        self.resnet_layer2 = nn.Sequential(*list(model.children())[5])
        self.resnet_layer3 = nn.Sequential(*list(model.children())[6])
        self.resnet_layer4 = nn.Sequential(*list(model.children())[7])

    def forward(self, x):
        x1 = self.resnet_layer1(x)
        x2 = self.resnet_layer2(x1)
        x3 = self.resnet_layer3(x2)
        x4 = self.resnet_layer4(x3)
        return x1, x2, x3, x4


class encode1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.resnet = nn.Sequential(*list(model.children())[0:8])

    def forward(self, x):
        x = self.resnet(x)
        return x


class OSnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = encode(model=resnet50(pretrained=True))
        self.encode1 = encode1(model=resnet50(pretrained=True))
        self.ca = ChannelAttention(in_planes=1024)
        # self.sa = SpatialAttention()
        self.encode_texture = Deep_Orientation(2048, 2048, 512)
        self.encode_texture1 = Deep_Orientation(2048, 2048, 512)
        self.embedding = nn.Sequential(
                nn.Conv2d(512, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                )

        self.decode2 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decode4 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 1),
        )


    def forward(self, image, patch):
        img1, img2, img3, img4 = self.encode(image)
        pat4 = self.encode1(patch)
        img4 = self.encode_texture(img4)
        pat4 = self.encode_texture1(pat4)


        for i in range(8):
            img_g = img4[:, 256 * i:256 * i + 256, :, :]
            pat = pat4[:, 256 * i:256 * i + 256, :, :]
            img_g = torch.cat([img_g, pat], dim=1)
            img_g = self.embedding(img_g)
            ca = self.ca(img_g)
            # if i == 7:
            #     caa = ca
            img_g = ca * img_g
            if i == 0:
                img = img_g
            else:
                img += img_g



        img = F.interpolate(img, 16)

        img = torch.cat([img, img3], dim=1)
        img = self.decode2(img)
        img = F.interpolate(img, 32)

        img = torch.cat([img, img2], dim=1)
        img = self.decode3(img)
        img = F.interpolate(img, 64)

        img = torch.cat([img, img1], dim=1)
        img = self.decode4(img)
        img = F.interpolate(img, 256)
        img = torch.sigmoid(img)
        return img

