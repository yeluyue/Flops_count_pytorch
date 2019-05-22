from torch.nn import  Module
import torch.nn.functional as F
import math
'''
Net work's common utility
'''

class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def Get_Conv_Size(height, width, kernel, stride, padding, rpt_num):
    conv_h = height
    conv_w = width
    for _ in range(rpt_num):
        conv_h = int((conv_h - kernel[0] + 2 * padding[0])/stride[0]+1)
        conv_w = int((conv_w - kernel[1] + 2 * padding[1])/stride[1]+1)
    return conv_h * conv_w

def Get_Conv_kernel(height, width, kernel, stride, padding, rpt_num):
    conv_h = height
    conv_w = width
    for _ in range(rpt_num):
        conv_h = math.ceil((conv_h - kernel[0] + 2 * padding[0])/stride[0]+1)
        conv_w = math.ceil((conv_w - kernel[1] + 2 * padding[1])/stride[1]+1)
        print(conv_h, conv_w)
    return (conv_h ,conv_w)


def get_dense_ave_pooling_size(height, width, block_config):
    size1 = Get_Conv_kernel(height, width, (3,3), (2,2), (1,1), 1)
    # print(size1)
    size2 = Get_Conv_kernel(size1[0], size1[1], (2, 2), (2, 2), (0, 0), len(block_config) )
    return size2

def get_shuffle_ave_pooling_size(height, width, using_pool):
    first_batch_num = 2
    if using_pool:
        first_batch_num = 3

    size1 = Get_Conv_kernel(height, width, (3,3), (2,2), (0,0), first_batch_num)
    # print(size1)
    size2 = Get_Conv_kernel(size1[0], size1[1], (2, 2), (2, 2), (0, 0), 2 )
    return size2

if __name__ == "__main__":
    get_dense_ave_pooling_size(112,112, [1,2,3,4])
    print("="*10)
    get_shuffle_ave_pooling_size(112,112,True)
    print("=" * 10)
    get_shuffle_ave_pooling_size(112, 112, False)
    print("=" * 10)