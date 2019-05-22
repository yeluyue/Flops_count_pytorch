from mxop.gluon import count_ops
from mxop.gluon import count_params
from mxop.gluon import op_summary
from models.mobilefacenet import MobileFaceNet
from models.shuffleNet_V2 import ShuffleNetV2



# model = resnet50()
# model = ShuffleNetV2()
model = MobileFaceNet(512)

op_counter = count_ops(model, input_size=(1, 3, 112, 112))
params_counter = count_params(model, input_size=(1, 3, 112, 112))

print('op_counter', op_counter)
# print('params_counter', params_counter)
