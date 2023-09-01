'''
Note: tUAPs transfer across input images, not models
'''
import torch
from torchvision.utils import save_image
import numpy as np
from torchvision import models
from torchvision.models import Inception_V3_Weights, ResNet50_Weights, DenseNet121_Weights, VGG16_BN_Weights
from utils_ImageNetCom import load_ground_truth, Normalize
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from torch.nn.parameter import Parameter

from attack_feature_FT_tUAP import FeatureFT
# customized networks: adding a few outputs to conventional networks
from net.resnet import ResNet18, ResNet50
from net.densenet import densenet121
from net.vgg16bn import vgg16_bn
from net.inception import inception_v3

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load model
checkpt_dir = 'C://Users/86188/.cache/torch/hub/checkpoints/'
model_1 = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=True).eval()
model_2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
model_3 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
model_4 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).eval()

for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False

model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

img_size = 299
batch_size = 2

input_path = 'tUAP/baseline/UAP_vgg16_CE/'

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])

# model = inception_v3(weights=None, transform_input=True)
# model.load_state_dict(torch.load(checkpt_dir + 'inception_v3_google-0cc3c7bd.pth'))

# model = ResNet50(num_classes=1000)
# model.load_state_dict(torch.load(checkpt_dir + 'resnet50-0676ba61.pth'))

# temp = torch.load(checkpt_dir + 'densenet121-a639ec97.pth')      # Densenet
# model = densenet121(weights=temp).eval()

temp = torch.load(checkpt_dir + 'vgg16_bn-6c64b313.pth')
model = vgg16_bn(weights=temp).eval()

########
model = model.to(device)
model.eval()

pos = 0
pos_ft = 0
torch.manual_seed(42)
for k in range(0, 2):
    if k % 1 == 0:
        print(k)
    X_adv = torch.zeros(batch_size, 3, img_size, img_size).to(device)
    X_cln = torch.zeros(batch_size, 3, img_size, img_size).to(device)
    labels_np = np.arange(batch_size) + k * batch_size
    labels = torch.tensor(labels_np, dtype=torch.int64).to(device)
    image_id = []
    for i in range(batch_size):
        label = labels_np[i]
        X_adv[i] = trn(Image.open(input_path + str(label) + '.png'))
        image_id.append(str(label))
    # UAP starts from a mean image
    X_cln = torch.full((batch_size, 3, img_size, img_size), 0.5).to(device)

    #### 2. feature space fine-tuning ####
    attack = FeatureFT(model=model, device=device, epsilon=16 / 255., k=10, model_name='vgg16_bn')
    # for UAP, we don't have the original label.
    X_adv_ft = attack.perturb(X_cln, X_adv, labels)

    for img_i in range(batch_size):
        save_image(X_adv_ft[img_i].data.cpu(), "tUAP/ft/UAP_vgg16_CE/" + image_id[img_i] + ".png")

    #### 3. verify  before fine-tune ####
    X_adv_norm = norm(X_adv).detach()
    output = model_4(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    print(output.gather(1, labels.unsqueeze(1)).sum().data)
    pos = pos + sum(predict_adv == labels).cpu().numpy()

    #### after fine-tune ####
    X_adv_norm = norm(X_adv_ft).detach()
    output = model_4(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    print(output.gather(1, labels.unsqueeze(1)).sum().data)
    pos_ft = pos_ft + sum(predict_adv2 == labels).cpu().numpy()

Done = 1
