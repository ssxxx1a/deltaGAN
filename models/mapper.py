import torch
import torch.nn as nn
from models.stylegan2.model_for_w_plus import EqualLinear


class Mapper(nn.Module):

    def __init__(self, in_channel=529, out_channel=512, n_layer=3):
        super(Mapper, self).__init__()
       # layers = [PixelNorm(), EqualLinear(in_channel, out_channel, lr_mul=1, activation='fused_lrelu')]
        layers = [EqualLinear(in_channel, out_channel, lr_mul=1, activation='fused_lrelu')]
        if n_layer > 1:
            for i in range(n_layer - 1):
                layers.append(EqualLinear(out_channel, out_channel, lr_mul=1, activation='fused_lrelu'))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x


class LevelsMapper(nn.Module):

    def __init__(self, input_c=529, n_layer=4):
        super(LevelsMapper, self).__init__()

        self.course_mapping = Mapper(in_channel=input_c, n_layer=n_layer)

        self.medium_mapping = Mapper(in_channel=input_c, n_layer=n_layer)

        self.fine_mapping = Mapper(in_channel=input_c, n_layer=n_layer)

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        x_coarse = self.course_mapping(x_coarse)

        x_medium = self.medium_mapping(x_medium)

        x_fine = self.fine_mapping(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

if __name__ == '__main__':
    from options.ddp_option import TrainOptions

    opts = TrainOptions().parse()
    model=LevelsMapper(opts, 17, n_layer=1).cuda()
    x=torch.rand(size=(1,14,17)).cuda()
    print(model(x))