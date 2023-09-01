from functools import partial
from typing import Union, List, Dict, Any, Optional, cast
import torch
import torch.nn as nn

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    #### add additional outputs
    def features_grad_multi_layers(self, x):
        # 3 299
        # x_l1 = self.conv1(x)
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)
        x_l1 = self.features[6](x)

        # x_l2 = self.conv2(x_l1)
        x_l2 = self.features[7](x_l1)
        x_l2 = self.features[8](x_l2)
        x_l2 = self.features[9](x_l2)
        x_l2 = self.features[10](x_l2)
        x_l2 = self.features[11](x_l2)
        x_l2 = self.features[12](x_l2)
        x_l2 = self.features[13](x_l2)
        x_l2.retain_grad()

        # x_l3 = self.conv3(x_l2)
        x_l3 = self.features[14](x_l2)
        x_l3 = self.features[15](x_l3)
        x_l3 = self.features[16](x_l3)
        x_l3 = self.features[17](x_l3)
        x_l3 = self.features[18](x_l3)
        x_l3 = self.features[19](x_l3)
        x_l3 = self.features[20](x_l3)
        x_l3 = self.features[21](x_l3)
        x_l3 = self.features[22](x_l3)
        x_l3 = self.features[23](x_l3)
        x_l3.retain_grad()

        # x_l4 = self.conv4(x_l3)
        x_l4 = self.features[24](x_l3)
        x_l4 = self.features[25](x_l4)
        x_l4 = self.features[26](x_l4)
        x_l4 = self.features[27](x_l4)
        x_l4 = self.features[28](x_l4)
        x_l4 = self.features[29](x_l4)
        x_l4 = self.features[30](x_l4)
        x_l4 = self.features[31](x_l4)
        x_l4 = self.features[32](x_l4)
        x_l4 = self.features[33](x_l4)
        x_l4.retain_grad()

        # x_l5 = self.conv5(x_l4)
        x_l5 = self.features[34](x_l4)
        x_l5 = self.features[35](x_l5)
        x_l5 = self.features[36](x_l5)
        x_l5 = self.features[37](x_l5)
        x_l5 = self.features[38](x_l5)
        x_l5 = self.features[39](x_l5)
        x_l5 = self.features[40](x_l5)
        x_l5 = self.features[41](x_l5)
        x_l5 = self.features[42](x_l5)
        x_l5 = self.features[43](x_l5)
        x_l5.retain_grad()

        fea = self.avgpool(x_l5)  # HZ

        fea = torch.flatten(fea, 1)
        out = self.classifier(fea)

        return out, x_l2, x_l3, x_l4, x_l5

    def multi_layer_features(self, x):
        # 3 299
        # x_l1 = self.conv1(x)
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)
        x_l1 = self.features[6](x)

        # x_l2 = self.conv2(x_l1)
        x_l2 = self.features[7](x_l1)
        x_l2 = self.features[8](x_l2)
        x_l2 = self.features[9](x_l2)
        x_l2 = self.features[10](x_l2)
        x_l2 = self.features[11](x_l2)
        x_l2 = self.features[12](x_l2)
        x_l2 = self.features[13](x_l2)

        # x_l3 = self.conv3(x_l2)
        x_l3 = self.features[14](x_l2)
        x_l3 = self.features[15](x_l3)
        x_l3 = self.features[16](x_l3)
        x_l3 = self.features[17](x_l3)
        x_l3 = self.features[18](x_l3)
        x_l3 = self.features[19](x_l3)
        x_l3 = self.features[20](x_l3)
        x_l3 = self.features[21](x_l3)
        x_l3 = self.features[22](x_l3)
        x_l3 = self.features[23](x_l3)

        # x_l4 = self.conv4(x_l3)
        x_l4 = self.features[24](x_l3)
        x_l4 = self.features[25](x_l4)
        x_l4 = self.features[26](x_l4)
        x_l4 = self.features[27](x_l4)
        x_l4 = self.features[28](x_l4)
        x_l4 = self.features[29](x_l4)
        x_l4 = self.features[30](x_l4)
        x_l4 = self.features[31](x_l4)
        x_l4 = self.features[32](x_l4)
        x_l4 = self.features[33](x_l4)

        # x_l5 = self.conv5(x_l4)
        x_l5 = self.features[34](x_l4)
        x_l5 = self.features[35](x_l5)
        x_l5 = self.features[36](x_l5)
        x_l5 = self.features[37](x_l5)
        x_l5 = self.features[38](x_l5)
        x_l5 = self.features[39](x_l5)
        x_l5 = self.features[40](x_l5)
        x_l5 = self.features[41](x_l5)
        x_l5 = self.features[42](x_l5)
        x_l5 = self.features[43](x_l5)

        return x_l2, x_l3, x_l4, x_l5

    ########


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
    "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
}


class VGG16_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138365992,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.360,
                    "acc@5": 91.516,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", VGG16_BN_Weights.IMAGENET1K_V1))
def vgg16_bn(*, weights: Optional[VGG16_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    .. autoclass:: torchvision.models.VGG16_BN_Weights
        :members:
    """
    weights = VGG16_BN_Weights.verify(weights)

    return _vgg("D", True, weights, progress, **kwargs)

