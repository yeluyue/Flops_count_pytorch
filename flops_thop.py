from torchvision.models import resnet18, resnet50
# from thop import profile
from profile_my import profile
# from models.mobilefacenet import MobileFaceNet
from models.shuffleNet_V2 import ShuffleNetV2
from models.MobileNetV2 import MobileNetV2
from models.model_lib import FaceNet_20
from models.resnet import resnet18, resnet34
from models.mobilefacenet_add import MobileFaceNet, MobileFaceNet_22, MobileFaceNet_y2,MobileFaceNet_23,MobileFaceNet_24
from models.mobilefacenet_add import MobileFaceNet_y2_se, MobileFaceNet_y2_2



# model = resnet50()
# model = ShuffleNetV2()
# model = MobileFaceNet(512)
# model = resnet50()
# model = MobileNetV2(n_class=512)
# model = FaceNet_20()
# model = resnet18()
# model = resnet34()
# model = MobileFaceNet_24(512)
model = MobileFaceNet_y2_2(512)


flops, params = profile(model, input_size=(1, 3, 112, 112))

print(flops, params)