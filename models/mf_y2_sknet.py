from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, ReLU6, Dropout2d, Dropout, \
    AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
from torch import nn


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)
#######################################################################################################################
class Residual_sk(Module):
    def __init__(self, c, num_block):
        super(Residual_sk, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                SKUnit(c, c, WH=32, M=3, G=8, r=2))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        # d = 8
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH / stride))
        #
        self.gap = AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        print(fea_U.shape)
        fea_s = self.gap(fea_U).squeeze_()
        print(fea_s.shape)

        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=16):
        #
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        fea = self.feas(x)
        fea_sh = fea + self.shortcut(x)
        fea_relu = self.relu(fea_sh)
        return fea_relu

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

################################## mobilefacenet_y2 #############################################################
class mf_y2_sknet(Module):
    # flops: 0.9775978496 G params: 4.413984 M
    def __init__(self, embedding_size):
        super(mf_y2_sknet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual_sk(64, num_block=2)
        self.conv_23 = SKUnit(64, 128, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_3 = Residual_sk(128, num_block=8)
        self.conv_34 = SKUnit(128, 256, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_4 = Residual_sk(256, num_block=16)
        self.conv_45 = SKUnit(256, 512, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_5 = Residual_sk(512, num_block=4)
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_sep = Conv_block(512, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)

######################################################################################################################
class mf_y2_sknet_res(Module):
    # flops: 0.9462642688 G params: 3.439072 M
    def __init__(self, embedding_size):
        super(mf_y2_sknet_res, self).__init__()
        Ci = 64
        self.conv1 = Conv_block(3, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual(Ci, num_block=1, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = SKUnit(Ci, Ci*2, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_3 = Residual(Ci*2, num_block=4, groups=Ci*4, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = SKUnit(Ci*2, Ci*4, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_4 = Residual(Ci*4, num_block=8, groups=Ci*8, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = SKUnit(Ci*4, Ci*8, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_5 = Residual(Ci*8, num_block=2, groups=Ci*16, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(Ci*8, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)

########################################################################################################################
class mf_y2_res_sknet(Module):
    # flops: 0.9765501952 G params: 2.662272 M
    def __init__(self, embedding_size):
        super(mf_y2_res_sknet, self).__init__()
        Ci = 64
        self.conv1 = Conv_block(3, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual_sk(Ci, num_block=8)
        self.conv_23 = Depth_Wise(Ci, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual_sk(Ci, num_block=32)
        self.conv_34 = Depth_Wise(Ci, Ci*2, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual_sk(Ci*2, num_block=32)
        self.conv_45 = Depth_Wise(Ci*2, Ci*4, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual_sk(Ci*4, num_block=8)
        self.conv_6_sep = Conv_block(Ci*4, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


########################################################################################################################
class Depth_Wise_SE_R(Module):
    def __init__(self, in_c, out_c, r, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise_SE_R, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.semoduel = SEModule(groups, reduction=r)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.semoduel(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual_sk_WH_M_r(Module):
    def __init__(self, c, WH, M, r, num_block):
        super(Residual_sk_WH_M_r, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                SKUnit(c, c, WH=WH, M=M, G=8, r=r))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)

#######################################################################################################################
class mf_y2_SE8_res_sknet_M3_R16(Module):
    # flops: 0.9789959168 G params: 2.591104 M
    def __init__(self, embedding_size):
        super(mf_y2_SE8_res_sknet_M3_R16, self).__init__()
        Ci = 64
        WH_set = 32
        M_set = 3
        r_set = 16
        SE_set = 8
        self.conv1 = Conv_block(3, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual_sk_WH_M_r(Ci, WH=WH_set, M=M_set, r=r_set, num_block=4)
        self.conv_23 = Depth_Wise_SE_R(Ci, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, r=SE_set)
        self.conv_3 = Residual_sk_WH_M_r(Ci, WH=WH_set, M=M_set, r=r_set, num_block=32)
        self.conv_34 = Depth_Wise_SE_R(Ci, Ci*2, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, r=SE_set)
        self.conv_4 = Residual_sk_WH_M_r(Ci*2, WH=WH_set, M=M_set, r=r_set, num_block=32)
        self.conv_45 = Depth_Wise_SE_R(Ci*2, Ci*4, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, r=SE_set)
        self.conv_5 = Residual_sk_WH_M_r(Ci*4, WH=WH_set, M=M_set, r=r_set,num_block=4)
        self.conv_6_sep = Conv_block(Ci*4, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)

#######################################################################################################################
class mf_y2_SEall_res_sknet_M3_R16(Module):
    # flops: 0.9789959168 G params: 2.591104 M
    def __init__(self, embedding_size):
        super(mf_y2_SEall_res_sknet_M3_R16, self).__init__()
        Ci = 64
        WH_set = 32
        M_set = 3
        r_set = 16
        self.conv1 = Conv_block(3, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual_sk_WH_M_r(Ci, WH=WH_set, M=M_set, r=r_set, num_block=4)
        self.conv_23 = Depth_Wise_SE_R(Ci, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, r=128)
        self.conv_3 = Residual_sk_WH_M_r(Ci, WH=WH_set, M=M_set, r=r_set, num_block=32)
        self.conv_34 = Depth_Wise_SE_R(Ci, Ci*2, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, r=256)
        self.conv_4 = Residual_sk_WH_M_r(Ci*2, WH=WH_set, M=M_set, r=r_set, num_block=32)
        self.conv_45 = Depth_Wise_SE_R(Ci*2, Ci*4, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, r=512)
        self.conv_5 = Residual_sk_WH_M_r(Ci*4, WH=WH_set, M=M_set, r=r_set,num_block=4)
        self.conv_6_sep = Conv_block(Ci*4, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)

######################################################################################################################
class mf_y2_res_sknet_M3(Module):
    # flops: 0.9765501952 G params: 2.662272 M
    def __init__(self, embedding_size):
        super(mf_y2_res_sknet_M3, self).__init__()
        Ci = 64
        self.conv1 = Conv_block(3, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual_sk(Ci, num_block=4)
        self.conv_23 = Depth_Wise(Ci, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual_sk(Ci, num_block=32)
        self.conv_34 = Depth_Wise(Ci, Ci*2, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual_sk(Ci*2, num_block=32)
        self.conv_45 = Depth_Wise(Ci*2, Ci*4, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual_sk(Ci*4, num_block=4)
        self.conv_6_sep = Conv_block(Ci*4, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


##############################################add se model to mobileface_y2############################################
class Depth_Wise_SE(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise_SE, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.semoduel = SEModule(out_c, 8)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.semoduel(x)
            output = short_cut + x
        else:
            output = x
        return output
class Residual_SE(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual_SE, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise_SE(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
######################################################################################################################
class mf_y2_sknet_res_se8(Module):
    # flops: 0.9469714432 G params: 3.575776 M
    def __init__(self, embedding_size):
        super(mf_y2_sknet_res_se8, self).__init__()
        Ci = 64
        self.conv1 = Conv_block(3, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual_SE(Ci, num_block=1, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = SKUnit(Ci, Ci*2, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_3 = Residual_SE(Ci*2, num_block=2, groups=Ci*4, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = SKUnit(Ci*2, Ci*4, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_4 = Residual_SE(Ci*4, num_block=4, groups=Ci*8, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = SKUnit(Ci*4, Ci*8, WH=32, M=2, G=8, r=2, stride=2)
        self.conv_5 = Residual_SE(Ci*8, num_block=1, groups=Ci*16, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(Ci*8, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)

######################################################################################################################
class mf_y2_sknet_M3_R16_res_SE8(Module):
    # flops: 0.8703602688 G params: 3.093568 M
    def __init__(self, embedding_size):
        super(mf_y2_sknet_M3_R16_res_SE8, self).__init__()
        G_set = 32 # 将skconv中间更改成Dwconv后，该参数失效
        Ci = 32
        m_set = 3
        r_set = 16
        epd = 2
        self.conv1 = Conv_block(3, Ci, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual_SE(Ci, num_block=2, groups=Ci * epd, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = SKUnit(Ci, Ci * 2, WH=32, M=m_set, G=G_set, r=r_set, stride=2)
        self.conv_3 = Residual_SE(Ci * 2, num_block=8, groups=Ci * 2 * epd, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = SKUnit(Ci * 2, Ci * 4, WH=32, M=m_set, G=G_set, r=r_set, stride=2)
        self.conv_4 = Residual_SE(Ci * 4, num_block=16, groups=Ci * 4 * epd, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = SKUnit(Ci * 4, Ci * 8, WH=32, M=m_set, G=G_set, r=r_set, stride=2)
        self.conv_5 = Residual_SE(Ci * 8, num_block=4, groups=Ci * 8 * epd, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(Ci * 8, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)