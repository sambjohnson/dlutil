import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as torchmodels

from .utils import *


class MRIResNet(nn.Module):
    """ ResNet18 model with (optional) user-specified number
        of final layers, as well as nonlinearity (must be from F).
        Pretrained.

        Optional number of unfrozen final blocks as well,
        from 0 to 6. Each of the 4 layers has 2 blocks, hence
        6 blocks unfreezes all but the first layer.

        Note that unfreezing occurs from bottom (output) of net
        towards top; original layers (edge detectors, etc.)
        probably do not / should not be finetuned.
    """

    def __init__(self, final_fcs=None, final_nonlin=None):
        if final_fcs is not None:
            assert isinstance(final_fcs, list) and len(final_fcs) > 0
        else:
            final_fcs = [32]
        super().__init__()
        self.model = torchmodels.resnet18(pretrained=True)
        nftr = self.model.fc.in_features  # number of original final layer nin
        self.model.fc = nn.Linear(nftr, final_fcs[0])
        self.fcs = nn.ModuleList()  # required to properly register params (functions like list)
        final_fcs += [1]
        for i in range(len(final_fcs[:-1])):
            self.fcs.append(nn.Linear(final_fcs[i], final_fcs[i + 1]))
        if final_nonlin is None:
            # defaults to elu
            self.nonlin = F.elu
        else:
            self.nonlin = final_nonlin

    def forward(self, x):
        x = self.nonlin(self.model(x))
        for layer in self.fcs:
            x = self.nonlin(layer(x))
        return x


class MNISTResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchmodels.resnet18(pretrained=True)
        nftr = self.model.fc.in_features  # number of original final layer nin
        self.model.fc = nn.Linear(nftr, 32, bias=True)
        self.fc2 = nn.Linear(32, 1, bias=True)

    def forward(self, x):
        x = F.elu(self.model(x))
        x = F.elu(self.fc2(x))
        return x


class ConvPoolBlock(nn.Module):
    """ Helper building block for larger networks.
        Two convolutional layers followed by an optional pooling layer.
        Pooling can also, optionally, be learnable by the second conv layer.
    """

    def __init__(self, cin, cmid, cout, k=4, downstride=2, pool='Max'):
        """ Two convolutional layers and one optional pooling layer.
            Args:
                cin, cmid, cout: three channels of two conv layers (cmid shared)
                k: kernel for all three layers
                downstride: the stride for downsampling
                pool: 'Max', 'Avg' or 'Conv'; if 'Conv', pooling layer is disabled
                    and the second Conv layer is set to downsample via stride=downstride
                    (note that this requires changing padding type to maintain correct
                    dimensions).
            Returns:
                An instance of a ConvPoolBlock. 
        """
        super().__init__()
        self.pad2 = 'same'
        if pool == 'Conv':
            self.c2stride = downstride
            self.pad2 = 0  # W_out = (W_in-K+2P)/S+1
        else:
            self.c2stride = 1
        self.c1 = nn.Conv2d(cin, cmid, k, padding='same')
        self.c2 = nn.Conv2d(cmid, cout, k, stride=self.c2stride, padding=self.pad2)
        if pool == 'Max':
            self.p = nn.MaxPool2d(k, stride=downstride)
        elif pool == 'Avg':
            self.p = nn.AvgPool2d(k, stride=downstride)
        else:
            self.p = None

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        if self.p is not None:
            x = self.p(x)
        return x


class MNISTConvNet(nn.Module):
    """ Class for small from-scratch cnn.
        Model dimensions are appropriate for 28x28 MNIST.
        For testing purposes.
    """

    def __init__(self):
        super().__init__()
        self.block1 = ConvPoolBlock(1, 4, 16)
        self.block2 = ConvPoolBlock(16, 32, 32)
        self.fc1 = nn.Linear(800, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SmallConvNet(nn.Module):
    """ Small conv net. Suitable for 200 x 200 images
        of type encountered in samjohns' sMRI work.
    """

    def __init__(self, pool='Max'):
        super().__init__()
        self.block1 = ConvPoolBlock(1, 32, 64, pool=pool)
        self.block2 = ConvPoolBlock(64, 128, 128, pool=pool)
        self.block3 = ConvPoolBlock(128, 128, 128, pool=pool)
        self.block4 = ConvPoolBlock(128, 128, 64, pool=pool)
        self.fc1 = nn.Linear(6400, 100)
        self.fc2 = nn.Linear(100, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        return x
