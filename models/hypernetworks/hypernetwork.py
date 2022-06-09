from torch import nn
from torch.nn import BatchNorm2d, PReLU, Sequential, Module
from torchvision.models import resnet34
import torch
from models.hypernetworks.refinement_blocks import HyperRefinementBlock, RefinementBlock, RefinementBlockSeparable
from models.hypernetworks.shared_weights_hypernet import SharedWeightsHypernet
from models.mapper import LevelsMapper, Mapper


class SharedWeightsHyperNetResNet(Module):

    def __init__(self, opts):
        super(SharedWeightsHyperNetResNet, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)
        self.aus_mapper = Mapper(in_channel=17, out_channel=256, n_layer=4)
        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        if len(opts.layers_to_tune) == 0:
            self.layers_to_tune = list(range(opts.n_hypernet_outputs))
        else:
            self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

        self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12]
        self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)

        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = opts.n_hypernet_outputs
        for layer_idx in range(self.n_outputs):
            if layer_idx in self.layers_to_tune:
                if layer_idx in self.shared_layers:
                    # 把conditions 直接加进来试试，所以这里channel变成513
                    refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512 + 1,
                                                            inner_c=128)
                else:
                    refinement_block = RefinementBlock(layer_idx, opts, n_channels=512 + 1, inner_c=256)
            else:
                refinement_block = None
            self.refinement_blocks.append(refinement_block)

    def forward(self, x, aus):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        weight_deltas = []
        delta_aus = self.aus_mapper(aus)
        delta_aus = delta_aus.view(aus.size(0), -1, 16, 16)
        # delta_aus=delta_aus.unsqueeze(dim=-1).unsqueeze(dim=-1)
        # delta_aus=delta_aus.repeat(1,1,16,16)
        # print(delta_aus.size())
        # print(x.size())
        x = torch.cat([x, delta_aus], dim=1)
        for j in range(self.n_outputs):
            if self.refinement_blocks[j] is not None:
                delta = self.refinement_blocks[j](x)
            else:
                delta = None
            weight_deltas.append(delta)
        return weight_deltas


class SharedWeightsHyperNetResNetSeparable(Module):

    def __init__(self, opts):
        super(SharedWeightsHyperNetResNetSeparable, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        if len(opts.layers_to_tune) == 0:
            self.layers_to_tune = list(range(opts.n_hypernet_outputs))
        else:
            self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

        self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12]
        self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)

        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = opts.n_hypernet_outputs
        for layer_idx in range(self.n_outputs):
            if layer_idx in self.layers_to_tune:
                if layer_idx in self.shared_layers:
                    refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
                else:
                    refinement_block = RefinementBlockSeparable(layer_idx, opts, n_channels=512, inner_c=256)
            else:
                refinement_block = None
            self.refinement_blocks.append(refinement_block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        weight_deltas = []
        for j in range(self.n_outputs):
            if self.refinement_blocks[j] is not None:
                delta = self.refinement_blocks[j](x)
            else:
                delta = None
            weight_deltas.append(delta)
        return weight_deltas
