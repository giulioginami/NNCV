import torch.nn as nn
import segmentation_models_pytorch as smp


def enable_dropout(model: nn.Module) -> None:
    """
    Re-enable all Dropout2d layers after model.eval() for MC Dropout.

    Call immediately after model.eval():
        model.eval()
        enable_dropout(model)

    Keeps BatchNorm in eval mode (uses running stats) while Dropout stays
    stochastic — the correct setup for Monte Carlo Dropout inference.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


class Model(nn.Module):
    """
    U-Net with a pretrained ResNet-34 encoder and MC Dropout in the decoder.

    A Dropout2d layer is appended after each decoder block's second conv.
    During training these layers act as standard regularisation.  At inference
    time, calling enable_dropout(model) after model.eval() keeps them active
    for Monte Carlo Dropout uncertainty estimation.

    nn.Dropout2d has no learnable parameters, so adding it does NOT change
    the state-dict key layout — existing checkpoints load without modification.
    """
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 19,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.net = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=n_classes,
        )

        # Append Dropout2d after every decoder block's second conv.
        # smp.Unet decoder has 5 DecoderBlock objects (one per skip-connection
        # level).  Each block exposes a .conv2 attribute (Conv2dReLU).
        for block in self.net.decoder.blocks:
            block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=dropout_p))

    def forward(self, x):
        return self.net(x)
