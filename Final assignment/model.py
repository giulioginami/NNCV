import torch.nn as nn
import segmentation_models_pytorch as smp


class Model(nn.Module):
    """
    U-Net with a pretrained ResNet-34 encoder (ImageNet weights).
    Built with segmentation_models_pytorch.

    """
    def __init__(
        self,
        in_channels=3,
        n_classes=19,
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
        """
        super().__init__()
        self.net = smp.Unet(
            encoder_name='resnet34',       # ResNet-34 backbone
            encoder_weights='imagenet',    # pretrained on ImageNet
            in_channels=in_channels,
            classes=n_classes,
        )

    def forward(self, x):
        return self.net(x)
