from torchvision.models import resnet18, resnet50
# from thop import profile
from profile_my import profile
# from models.mobilefacenet import MobileFaceNet
from models.shuffleNet_V2 import ShuffleNetV2
from models.MobileNetV2 import MobileNetV2
from models.model_lib import FaceNet_20
from models.resnet import resnet18, resnet34
from models.mobilefacenet_add import MobileFaceNet, MobileFaceNet_22, MobileFaceNet_y2,MobileFaceNet_23,MobileFaceNet_24
from models.mobilefacenet_add import MobileFaceNet_y2_se, MobileFaceNet_y2_2, MobileFaceNet_sor, MobileFaceNet_y2_3
from models.mobilefacenet_add import MobileFaceNet_y2_4, MobileFaceNet_y2_5, MobileFaceNet_y2_6, mf_y2_mbv2_t6
from models.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large, MobileNetV3_Large_ex2, MobileNetV3_Large_epx2

from models.sknet import SKNet
from models.mf_y2_sknet import mf_y2_sknet, mf_y2_sknet_res, mf_y2_res_sknet, mf_y2_sknet_res_se8, mf_y2_sknet_res_M3



# model = resnet50()
# model = ShuffleNetV2()
# model = MobileFaceNet(512)
# model = resnet50()
# model = MobileNetV2(n_class=512)
# model = FaceNet_20()
# model = resnet18()
# model = resnet34()
# model = MobileFaceNet_24(512)
# model = MobileFaceNet_y2(512)
# model = MobileNetV3_Small(512)
# model = MobileNetV3_Large(512)
# model = MobileFaceNet_sor(512)
# model = MobileFaceNet_y2_6(512)

# model = mf_y2_mbv2_t6(512)


# model = MobileNetV3_Large_ex2(512)
# model = MobileNetV3_Large_epx2(512)

# model = SKNet(512)

# model = mf_y2_sknet(512)
# model = mf_y2_sknet_res(512)
# model = mf_y2_res_sknet(512)

model = mf_y2_sknet_res_M3(512)


# model = mf_y2_sknet_res_se8(512)

flops, params = profile(model, input_size=(10, 3, 112, 112))
#
print('flops:', (flops/(1e10)),'G', 'params:', params/(1e6),'M')
print('./.....................................................................................................................................................................')
# print(model)
