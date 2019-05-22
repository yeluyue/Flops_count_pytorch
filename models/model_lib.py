'''
Modify self defination net Work
'''
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
from models.model import Conv_block
from models.common_utility import L2Norm, Flatten, Get_Conv_Size


class Conv_Relu_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_Relu_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.prelu = PReLU(out_c, init=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class FaceNet_Res_Elem(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super(FaceNet_Res_Elem, self).__init__()
        self.conv1 = Conv_block(in_c, out_c=in_c, kernel=kernel, padding=padding, stride=stride, groups=groups)
        self.conv2 = Conv_block(in_c, out_c=out_c, kernel=kernel, padding=padding, stride=stride, groups=groups)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.residual:
            output = x + short_cut
        else:
            output = x

        return output

class FaceNet_Res_Elem_conv_relu(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super(FaceNet_Res_Elem_conv_relu, self).__init__()
        self.conv1 = Conv_Relu_block(in_c, out_c=in_c, kernel=kernel, padding=padding, stride=stride, groups=groups)
        self.conv2 = Conv_Relu_block(in_c, out_c=out_c, kernel=kernel, padding=padding, stride=stride, groups=groups)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.residual:
            output = x + short_cut
        else:
            output = x

        return output

class Common_Residual(Module):
    def __init__(self, Res_Elem, in_c, out_c, num, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super(Common_Residual, self).__init__()
        modules = []
        for _ in range(num):
            modules.append(Res_Elem(in_c, out_c, True, kernel=kernel, stride=stride, padding=padding, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        output = self.model(x)
        return output


#================================================FaceNet-20=============================================================

class FaceNet_20(Module):
    def __init__(self, embedding_size = 512, height = 112, width = 112):
        super(FaceNet_20, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3,3), stride=(2,2), padding=(1,1), groups=1)
        self.conv1_res = Common_Residual(FaceNet_Res_Elem, 64,64,1, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv2 = Conv_block(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv2_res = Common_Residual(FaceNet_Res_Elem, 128,128, 2, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv3 = Conv_block(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv3_res = Common_Residual(FaceNet_Res_Elem, 256,256, 4, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv4 = Conv_block(256, 512, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv4_res = Common_Residual(FaceNet_Res_Elem, 512,512, 1, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.flatten = Flatten()
        self.linear = Linear(
            512 * Get_Conv_Size(height, width, kernel=(3,3), stride=(2,2), padding=(1,1), rpt_num=4),
            embedding_size)
        self.bn = BatchNorm1d(embedding_size)
        self.l2 = L2Norm()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_res(out)
        out = self.conv2(out)
        out = self.conv2_res(out)
        out = self.conv3(out)
        out = self.conv3_res(out)
        out = self.conv4(out)
        out = self.conv4_res(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        out = self.l2(out)
        return out


class FaceNet_Origin_20(Module):
    def __init__(self, embedding_size = 512, height = 112, width = 96):
        super(FaceNet_Origin_20, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3,3), stride=(2,2), padding=(1,1), groups=1)
        self.conv1_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 64,64,1, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv2 = Conv_block(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv2_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 128,128, 2, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv3 = Conv_block(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv3_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 256,256, 4, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv4 = Conv_block(256, 512, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv4_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 512,512, 1, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.flatten = Flatten()
        self.linear = Linear(
            512* Get_Conv_Size(height, width, kernel=(3,3), stride=(2,2), padding=(1,1), rpt_num=4),
            embedding_size)
        self.bn = BatchNorm1d(embedding_size)
        self.l2 = L2Norm()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_res(out)
        out = self.conv2(out)
        out = self.conv2_res(out)
        out = self.conv3(out)
        out = self.conv3_res(out)
        out = self.conv4(out)
        out = self.conv4_res(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        out = self.l2(out)
        return out


class FaceNet_cf_20(Module):
    def __init__(self, embedding_size = 512, height = 112, width = 96):
        super(FaceNet_cf_20, self).__init__()
        self.conv1 = Conv_block_no_bn(3, 64, kernel=(3,3), stride=(2,2), padding=(1,1), groups=1)
        self.conv1_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 64,64,1, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv2 = Conv_block(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv2_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 128,128, 2, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv3 = Conv_block_no_bn(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv3_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 256,256, 4, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.conv4 = Conv_block_no_bn(256, 512, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
        self.conv4_res = Common_Residual(FaceNet_Res_Elem_conv_relu, 512,512, 1, kernel=(3,3), stride=(1,1), padding=(1,1), groups=1)
        self.flatten = Flatten()
        self.linear = Linear(
            512* Get_Conv_Size(height, width, kernel=(3,3), stride=(2,2), padding=(1,1), rpt_num=4),
            embedding_size)
        self.bn = BatchNorm1d(embedding_size)
        self.l2 = L2Norm()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_res(out)
        out = self.conv2(out)
        out = self.conv2_res(out)
        out = self.conv3(out)
        out = self.conv3_res(out)
        out = self.conv4(out)
        out = self.conv4_res(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        out = self.l2(out)
        return out


