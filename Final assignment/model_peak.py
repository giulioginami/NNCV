import torch.nn as nn
import segmentation_models_pytorch as smp


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 19,
    ):
        super().__init__()
        self.net = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=n_classes,
        )

    def forward(self, x):
        return self.net(x)
