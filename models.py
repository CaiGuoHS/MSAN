import torch
import torch.nn as nn

def conv(in_planes, out_planes, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=(kernel_size//2), bias=bias)

def convT(in_planes, out_planes, kernel_size, stride=1, padding=1, bias=True):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=bias)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.gconv = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=3, stride=2, padding=1, groups = self.chanel_in)
        self.query_conv = conv(self.chanel_in, self.chanel_in, 1)
        self.key_conv = conv(self.chanel_in, self.chanel_in, 1)
        self.value_conv = conv(self.chanel_in, self.chanel_in, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.degconv = nn.ConvTranspose2d(self.chanel_in, self.chanel_in, kernel_size=4, stride=2, padding=1, groups = self.chanel_in)

    def forward(self, in_x):
        x = self.gconv(in_x)
        m_batchsize, C, width, height = x.size()
        wh = width * height

        proj_query = self.query_conv(x).view(m_batchsize, -1, wh).permute(0, 2, 1)  # B X C X N --> B X N X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, wh)  # B X C x N
        energy = torch.bmm(proj_query, proj_key)  # transpose check N X N
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, wh)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.degconv(out)
        out = self.gamma * out + in_x
        return out

class ResBlock(nn.Module):
    def __init__(self, in_planes=32, out_planes=32):
        super().__init__()
        self.conv1 = conv(in_planes, out_planes, 3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(in_planes, out_planes, 3)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return identity + x

class EDcoder(nn.Module):
    def __init__(self, level, num_resblocks, input_channels):
        super().__init__()
        # Conv
        self.layer1 = conv(input_channels, 32, kernel_size=3)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(32, 32))
        self.layer2 = nn.Sequential(*modules)
        # Conv
        self.layer3 = conv(32, 64, kernel_size=3, stride=2)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(64, 64))
        self.layer4 = nn.Sequential(*modules)
        # Conv
        self.layer5 = conv(64, 128, kernel_size=3, stride=2)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(128, 128))
        self.layer6 = nn.Sequential(*modules)
        self.sa = Self_Attn(128)
        # Deconv
        self.layer7 = convT(128, 64, kernel_size=4, stride=2)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(64, 64))
        self.layer8 = nn.Sequential(*modules)
        # Deconv
        self.layer9 = convT(64, 32, kernel_size=4, stride=2)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(32, 32))
        self.layer10 = nn.Sequential(*modules)
        # Conv
        self.layer11 = conv(32, 3, kernel_size=3)

    def forward(self, x, f=0):
        # Conv
        x = self.layer1(x)
        x2 = self.layer2(x)
        # Conv
        x = self.layer3(x2)
        x4 = self.layer4(x)
        # Conv
        x = self.layer5(x4)
        x = x + f
        x6 = self.layer6(x)
        x = self.sa(x6)
        # Deconv
        x = self.layer7(x)
        x = x + x4
        x = self.layer8(x)
        # Deconv
        x = self.layer9(x)
        x = x + x2
        x = self.layer10(x)
        # Conv
        x = self.layer11(x)
        return x, x6

class model(nn.Module):
    def __init__(self, num_resblocks, input_channels):
        super().__init__()
        self.edcoder1 = EDcoder(level=1, num_resblocks=num_resblocks[0], input_channels=input_channels[0])
        self.edcoder2 = EDcoder(level=2, num_resblocks=num_resblocks[1], input_channels=input_channels[1])
        self.edcoder3 = EDcoder(level=3, num_resblocks=num_resblocks[2], input_channels=input_channels[2])
        self.edcoder4 = EDcoder(level=4, num_resblocks=num_resblocks[3], input_channels=input_channels[3])

    def forward(self, x):
        d1, f1 = self.edcoder1(x)
        d2, f2 = self.edcoder2(torch.cat((x, d1), 1), f1)
        d3, f3 = self.edcoder3(torch.cat((x, d2), 1), f2)
        d4 = self.edcoder4(torch.cat((x, d3), 1), f3)[0]

        return d4, d3, d2, d1