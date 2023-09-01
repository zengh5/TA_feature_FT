import torch
import torch.nn as nn
import numpy as np
from utils_ImageNetCom import Normalize, FIAloss, DI, DI_keepresolution, gkern
from matplotlib import pyplot as plt
import torch.nn.functional as F

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()


class FeatureFT(object):
    # imageNet
    def __init__(self, model=None, device=None, epsilon=16 / 255., k=10, alpha=1.6 / 255., prob=0.7,
                 mask_num=30, mu=1.0, model_name='res18'):
        # set Parameters
        self.model = model.to(device)
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.prob = prob          # for normal model, drop 0.3; for defense models, drop 0.1
        self.mask_num = mask_num  # according to paper 30
        self.mu = mu
        self.device = device
        self.model_name = model_name

    '''
    X_nat: the original image;
    X_adv: an AE crafted by a baseline attack;
    y_tar: target label
    y_ori: the original label (of X_nat)
    '''
    def perturb(self, X_nat, X_adv, y_tar, y_ori):
        self.alpha = self.epsilon / 16.
        # get grads
        labels_tar = y_tar.clone().detach().to(self.device)
        labels_ori = y_ori.clone().detach().to(self.device)
        # in place
        _, temp_x_l1, temp_x_l2, temp_x_l3, temp_x_l4 = self.model.features_grad_multi_layers(X_nat)

        batch_size = X_nat.shape[0]
        image_size = X_nat.shape[-1]

        # calculate the feature importance (to y_o) from the clean image
        grad_sum_l1 = torch.zeros(temp_x_l1.shape).to(device)
        grad_sum_l2 = torch.zeros(temp_x_l2.shape).to(device)
        grad_sum_l3 = torch.zeros(temp_x_l3.shape).to(device)
        grad_sum_l4 = torch.zeros(temp_x_l4.shape).to(device)
        for i in range(self.mask_num):
            self.model.zero_grad()
            img_temp_i = norm(X_nat).clone()
            # drop a few pixels randomly
            mask = torch.tensor(np.random.binomial(1, self.prob, size=(batch_size, 3, image_size, image_size))).to(
                device)
            img_temp_i = img_temp_i * mask
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            # gather logit_o
            logit_label = logits.gather(1, labels_ori.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()
            # aggregate the gradient
            grad_sum_l1 += x_l1.grad
            grad_sum_l2 += x_l2.grad
            grad_sum_l3 += x_l3.grad
            grad_sum_l4 += x_l4.grad

        # normalize the aggregated gradient. You can change it to average
        grad_sum_l1 = grad_sum_l1 / grad_sum_l1.std()
        grad_sum_l2 = grad_sum_l2 / grad_sum_l2.std()
        grad_sum_l3 = grad_sum_l3 / grad_sum_l3.std()
        grad_sum_l4 = grad_sum_l4 / grad_sum_l4.std()

        # calculate the feature importance from an AE crafted by a baseline attack
        grad_sum_mid_l1 = torch.zeros(temp_x_l1.shape).to(device)
        grad_sum_mid_l2 = torch.zeros(temp_x_l2.shape).to(device)
        grad_sum_mid_l3 = torch.zeros(temp_x_l3.shape).to(device)
        grad_sum_mid_l4 = torch.zeros(temp_x_l4.shape).to(device)
        for i in range(self.mask_num):
            self.model.zero_grad()
            img_temp_i = norm(X_adv).clone()
            mask = torch.tensor(np.random.binomial(1, self.prob, size=(batch_size, 3, image_size, image_size))).to(
                device)
            img_temp_i = img_temp_i * mask
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            # gather logit_t
            logit_label = logits.gather(1, labels_tar.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()

            grad_sum_mid_l1 += x_l1.grad
            grad_sum_mid_l2 += x_l2.grad
            grad_sum_mid_l3 += x_l3.grad
            grad_sum_mid_l4 += x_l4.grad

        # normalize the aggregated gradient. You can change it to average
        grad_sum_mid_l1 = grad_sum_mid_l1 / grad_sum_mid_l1.std()
        grad_sum_mid_l2 = grad_sum_mid_l2 / grad_sum_mid_l2.std()
        grad_sum_mid_l3 = grad_sum_mid_l3 / grad_sum_mid_l3.std()
        grad_sum_mid_l4 = grad_sum_mid_l4 / grad_sum_mid_l4.std()

        # Eq.(6) Enhance the feature contribute to y_t and suppress that contribute to y_o
        beta = 0.2
        grad_sum_new_l1 = grad_sum_mid_l1 - beta * grad_sum_l1
        grad_sum_new_l2 = grad_sum_mid_l2 - beta * grad_sum_l2
        grad_sum_new_l3 = grad_sum_mid_l3 - beta * grad_sum_l3
        grad_sum_new_l4 = grad_sum_mid_l4 - beta * grad_sum_l4

        # Eq.(7), main loop
        g = 0
        x_cle = X_nat.detach()
        x_adv_ft = X_adv.clone().requires_grad_()
        for epoch in range(self.k):
            self.model.zero_grad()
            x_adv_ft.requires_grad_()
            x_adv_ft_DI = DI_keepresolution(x_adv_ft)                       # DI
            x_adv_norm = norm(x_adv_ft_DI)                                  # [0, 1] to [-1, 1]
            mid_feature_l1, mid_feature_l2, mid_feature_l3, mid_feature_l4 = self.model.multi_layer_features(x_adv_norm)

            # loss = FIAloss(grad_sum_new_l1, mid_feature_l1)
            # loss = FIAloss(grad_sum_new_l2, mid_feature_l2)
            loss = FIAloss(grad_sum_new_l3, mid_feature_l3)
            # loss = FIAloss(grad_sum_new_l4, mid_feature_l4)

            loss.backward()
            grad_c = x_adv_ft.grad
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
            g = self.mu * g + grad_c                                                                   # MI

            x_adv_ft = x_adv_ft + self.alpha * g.sign()
            with torch.no_grad():
                eta = torch.clamp(x_adv_ft - x_cle, min=-self.epsilon, max=self.epsilon)
                # X_ft = torch.clamp(x_cle + eta, min=0, max=1).detach_()
            x_adv_ft = torch.clamp(x_cle + eta, min=0, max=1)

        return x_adv_ft.detach()
