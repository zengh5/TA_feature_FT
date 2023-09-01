import torch
from torchvision.utils import save_image
import numpy as np
from torchvision import models
from torchvision.models import Inception_V3_Weights, ResNet50_Weights, DenseNet121_Weights, VGG16_BN_Weights
from attack_feature_FT import FeatureFT
from torchvision import transforms
import torch.nn as nn
from PIL import Image

from utils_ImageNetCom import load_ground_truth, Normalize
# customized networks: adding a few outputs to conventional networks
from net.resnet import ResNet18, ResNet50
from net.densenet import densenet121
from net.vgg16bn import vgg16_bn
from net.inception import inception_v3


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load model # load pretrained model，
# modify it to the dir of the pretrained model in your computer
checkpt_dir = 'C://Users/86188/.cache/torch/hub/checkpoints/'
# model_1 = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=True).eval()
model_2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
model_3 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
model_4 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).eval()

# for param in model_1.parameters():
#     param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False

# model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

img_size = 299
batch_size = 4
clean_path = 'E://Python/AE_transfer/Target/dataset/images/'  # clean images

adv_path = 'adv_imgs/CE/incV3/'   # Your AEs
# adv_path = '../github_transfer_target/adv_imgs/CE/incV3/'   # Your AEs

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
# trn = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

model = inception_v3(weights=None, transform_input=True)
model.load_state_dict(torch.load(checkpt_dir + 'inception_v3_google-0cc3c7bd.pth'))

# model = ResNet50(num_classes=1000)  # Hui Zeng
# model.load_state_dict(torch.load(checkpt_dir + 'resnet50-0676ba61.pth'))

# temp = torch.load(checkpt_dir + 'densenet121-a639ec97.pth')      # Densenet
# model = densenet121(weights=temp).eval()

# temp = torch.load(checkpt_dir + 'vgg16_bn-6c64b313.pth')
# model = vgg16_bn(weights=temp).eval()

########
model = model.to(device)
model.eval()

pos = np.zeros(4)
neg_ori = np.zeros(4)        # restored
pos_ft = np.zeros(4)
neg_ori_ft = np.zeros(4)     # restored

torch.manual_seed(42)
for k in range(0, 2):
    if k % 1 == 0:
        print(k)
    #### 1. preparing data ####
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_adv = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    X_cln = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    for i in range(batch_size_cur):
        X_adv[i] = trn(Image.open(adv_path + image_id_list[k * batch_size + i] + '.png'))
        X_cln[i] = trn(Image.open(clean_path + image_id_list[k * batch_size + i] + '.png'))
    labels_ori = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()
    # predefined random-target scenario
    labels_tar = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()

    # uncomment the following block for the low-rank target scenario
    # with torch.no_grad():
    #     logits_ori = model(norm(X_cln))
    #     _, out_class = torch.sort(logits_ori, dim=1, descending=True)
    # labels_tar = out_class[:, 999]
    #######

    #### 2. feature space fine-tuning ####
    attack = FeatureFT(model=model, device=device, epsilon=16 / 255., k=10)
    X_adv_ft = attack.perturb(X_cln, X_adv, labels_tar, labels_ori)

    #### 3. verify  before fine-tune ####
    X_adv_norm = norm(X_adv).detach()

    output = model(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos[0] = pos[0] + sum(predict_adv == labels_tar).cpu().numpy()
    neg_ori[0] = neg_ori[0] + sum(predict_adv == labels_ori).cpu().numpy()

    output = model_2(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos[1] = pos[1] + sum(predict_adv == labels_tar).cpu().numpy()

    output = model_3(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    # print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos[2] = pos[2] + sum(predict_adv == labels_tar).cpu().numpy()
    neg_ori[2] = neg_ori[2] + sum(predict_adv == labels_ori).cpu().numpy()

    output = model_4(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    # print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos[3] = pos[3] + sum(predict_adv == labels_tar).cpu().numpy()
    neg_ori[3] = neg_ori[3] + sum(predict_adv == labels_ori).cpu().numpy()

    #### after fine-tune ####
    X_adv_norm = norm(X_adv_ft).detach()

    output = model(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos_ft[0] = pos_ft[0] + sum(predict_adv2 == labels_tar).cpu().numpy()
    neg_ori_ft[0] = neg_ori_ft[0] + sum(predict_adv2 == labels_ori).cpu().numpy()

    output = model_2(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos_ft[1] = pos_ft[1] + sum(predict_adv2 == labels_tar).cpu().numpy()

    output = model_3(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    # print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos_ft[2] = pos_ft[2] + sum(predict_adv2 == labels_tar).cpu().numpy()
    neg_ori_ft[2] = neg_ori_ft[2] + sum(predict_adv2 == labels_ori).cpu().numpy()

    output = model_4(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    # print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos_ft[3] = pos_ft[3] + sum(predict_adv2 == labels_tar).cpu().numpy()
    neg_ori_ft[3] = neg_ori_ft[3] + sum(predict_adv2 == labels_ori).cpu().numpy()

print(pos)
print(pos_ft)
print(neg_ori)
print(neg_ori_ft)
Done = 1
