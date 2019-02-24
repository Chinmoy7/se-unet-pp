import torch
import torch.nn as nn

class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):

        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):

        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor

class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):

        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=True):
    if useBN:
        return nn.Sequential(
          nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.ReLU(inplace=True),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
          nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU(inplace=True)
        )

def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

class SE_U_Net(nn.Module):
    def __init__(self, in_channels, out_channels, fc_dim=512, useBN=True, useCSE=False, useSSE=False, useCSSE=True):
        super(SE_U_Net, self).__init__()
        nb_filter = [fc_dim/16, fc_dim/8, fc_dim/4, fc_dim/2, fc_dim]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.useCSE = useCSE
        self.useSSE = useSSE
        self.useCSSE = useCSSE

        self.conv1 = add_conv_stage(self.in_channels, nb_filter[0], useBN=useBN)
        self.conv2 = add_conv_stage(nb_filter[0], nb_filter[1], useBN=useBN)
        self.conv3 = add_conv_stage(nb_filter[1], nb_filter[2], useBN=useBN)
        self.conv4 = add_conv_stage(nb_filter[2], nb_filter[3], useBN=useBN)
        self.conv5 = add_conv_stage(nb_filter[3], nb_filter[4], useBN=useBN)

        self.conv4m = add_conv_stage(nb_filter[3]+nb_filter[3], nb_filter[3], useBN=useBN)
        self.conv3m = add_conv_stage(nb_filter[2]+nb_filter[2], nb_filter[2], useBN=useBN)
        self.conv2m = add_conv_stage(nb_filter[1]+nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1m = add_conv_stage(nb_filter[0]+nb_filter[0], nb_filter[0], useBN=useBN)

        self.conv0 = nn.Conv2d(nb_filter[0], self.out_channels, 3, 1, 1)

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(nb_filter[4], nb_filter[3])
        self.upsample43 = upsample(nb_filter[3], nb_filter[2])
        self.upsample32 = upsample(nb_filter[2], nb_filter[1])
        self.upsample21 = upsample(nb_filter[1], nb_filter[0])

        self.cse1 = ChannelSELayer(nb_filter[0], 2)
        self.cse2 = ChannelSELayer(nb_filter[1], 2)
        self.cse3 = ChannelSELayer(nb_filter[2], 2)
        self.cse4 = ChannelSELayer(nb_filter[3], 2)

        self.cse4m = ChannelSELayer(nb_filter[3], 2)
        self.cse3m = ChannelSELayer(nb_filter[2], 2)
        self.cse2m = ChannelSELayer(nb_filter[1], 2)
        self.cse1m = ChannelSELayer(nb_filter[0], 2)

        self.sse1 = SpatialSELayer(nb_filter[0])
        self.sse2 = SpatialSELayer(nb_filter[1])
        self.sse3 = SpatialSELayer(nb_filter[2])
        self.sse4 = SpatialSELayer(nb_filter[3])

        self.sse4m = SpatialSELayer(nb_filter[3])
        self.sse3m = SpatialSELayer(nb_filter[2])
        self.sse2m = SpatialSELayer(nb_filter[1])
        self.sse1m = SpatialSELayer(nb_filter[0])

        self.csse1 = ChannelSpatialSELayer(nb_filter[0], 2)
        self.csse2 = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse3 = ChannelSpatialSELayer(nb_filter[2], 2)
        self.csse4 = ChannelSpatialSELayer(nb_filter[3], 2)

        self.csse4m = ChannelSpatialSELayer(nb_filter[3], 2)
        self.csse3m = ChannelSpatialSELayer(nb_filter[2], 2)
        self.csse2m = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse1m = ChannelSpatialSELayer(nb_filter[0], 2)


    def forward(self, x):
        if(self.useCSSE):
            conv1_ = self.csse1(self.conv1(x))
            conv2_ = self.csse2(self.conv2(self.max_pool(conv1_)))
            conv3_ = self.csse3(self.conv3(self.max_pool(conv2_)))
            conv4_ = self.csse4(self.conv4(self.max_pool(conv3_)))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.csse4m(self.conv4m(conv5_))

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.csse3m(self.conv3m(conv4_))

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.csse2m(self.conv2m(conv3_))

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.csse1m(self.conv1m(conv2_))

            conv0_ = self.conv0(conv1_)

        elif(self.useCSE):
            conv1_ = self.cse1(self.conv1(x))
            conv2_ = self.cse2(self.conv2(self.max_pool(conv1_)))
            conv3_ = self.cse3(self.conv3(self.max_pool(conv2_)))
            conv4_ = self.cse4(self.conv4(self.max_pool(conv3_)))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.cse4m(self.conv4m(conv5_))

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.cse3m(self.conv3m(conv4_))

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.cse2m(self.conv2m(conv3_))

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.cse1m(self.conv1m(conv2_))

            conv0_ = self.conv0(conv1_)

        elif(self.useSSE):
            conv1_ = self.sse1(self.conv1(x))
            conv2_ = self.sse2(self.conv2(self.max_pool(conv1_)))
            conv3_ = self.sse3(self.conv3(self.max_pool(conv2_)))
            conv4_ = self.sse4(self.conv4(self.max_pool(conv3_)))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.sse4m(self.conv4m(conv5_))

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.sse3m(self.conv3m(conv4_))

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.sse2m(self.conv2m(conv3_))

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.sse1m(self.conv1m(conv2_))

            conv0_ = self.conv0(conv1_)

        else:
            conv1_ = self.conv1(x)
            conv2_ = self.conv2(self.max_pool(conv1_))
            conv3_ = self.conv3(self.max_pool(conv2_))
            conv4_ = self.conv4(self.max_pool(conv3_))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.conv4m(conv5_)

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.conv3m(conv4_)

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.conv2m(conv3_)

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.conv1m(conv2_)

            conv0_ = self.conv0(conv1_)            

        return conv0_

class SE_U_Net_PP(nn.Module):
    def __init__(self, in_channels, out_channels, fc_dim=512, useBN=True, useCSE=False, useSSE=False, useCSSE=True):
        super(SE_U_Net_PP, self).__init__()
        nb_filter = [fc_dim//16, fc_dim//8, fc_dim//4, fc_dim//2, fc_dim]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.useCSE = useCSE
        self.useSSE = useSSE
        self.useCSSE = useCSSE

        self.conv1_1   = add_conv_stage(self.in_channels, nb_filter[0], useBN=useBN)
        self.conv2_1   = add_conv_stage(nb_filter[0], nb_filter[1], useBN=useBN)
        self.conv1_2   = add_conv_stage(nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)
        self.conv3_1   = add_conv_stage(nb_filter[1], nb_filter[2], useBN=useBN)
        self.conv2_2   = add_conv_stage(nb_filter[1] + nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1_3   = add_conv_stage(nb_filter[0] + nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)
        self.conv4_1   = add_conv_stage(nb_filter[2], nb_filter[3], useBN=useBN)
        self.conv3_2   = add_conv_stage(nb_filter[2] + nb_filter[2], nb_filter[2], useBN=useBN)
        self.conv2_3   = add_conv_stage(nb_filter[1] + nb_filter[1] + nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1_4   = add_conv_stage(nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)
        self.conv5_1   = add_conv_stage(nb_filter[3], nb_filter[4], useBN=useBN)
        self.conv4_2   = add_conv_stage(nb_filter[3] + nb_filter[3], nb_filter[3], useBN=useBN)
        self.conv3_3   = add_conv_stage(nb_filter[2] + nb_filter[2] + nb_filter[2], nb_filter[2], useBN=useBN)
        self.conv2_4   = add_conv_stage(nb_filter[1] + nb_filter[1] + nb_filter[1] + nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1_5   = add_conv_stage(nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)

        self.conv0 = nn.Conv2d(nb_filter[0], self.out_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(4*self.out_channels, self.out_channels, 3, 1, 1)

        self.max_pool = nn.MaxPool2d(2)

        self.up1_2 = upsample(nb_filter[1], nb_filter[0])
        self.up2_2 = upsample(nb_filter[2], nb_filter[1])
        self.up1_3 = upsample(nb_filter[1], nb_filter[0])
        self.up3_2 = upsample(nb_filter[3], nb_filter[2])
        self.up2_3 = upsample(nb_filter[2], nb_filter[1])
        self.up1_4 = upsample(nb_filter[1], nb_filter[0])
        self.up4_2 = upsample(nb_filter[4], nb_filter[3])
        self.up3_3 = upsample(nb_filter[3], nb_filter[2])
        self.up2_4 = upsample(nb_filter[2], nb_filter[1])
        self.up1_5 = upsample(nb_filter[1], nb_filter[0])

        self.cse1_1 = ChannelSELayer(nb_filter[0], 2)
        self.cse2_1 = ChannelSELayer(nb_filter[1], 2)
        self.cse1_2 = ChannelSELayer(nb_filter[0], 2)
        self.cse3_1 = ChannelSELayer(nb_filter[2], 2)
        self.cse2_2 = ChannelSELayer(nb_filter[1], 2)
        self.cse1_3 = ChannelSELayer(nb_filter[0], 2)
        self.cse4_1 = ChannelSELayer(nb_filter[3], 2)
        self.cse3_2 = ChannelSELayer(nb_filter[2], 2)
        self.cse2_3 = ChannelSELayer(nb_filter[1], 2)
        self.cse1_4 = ChannelSELayer(nb_filter[0], 2)
        self.cse4_2 = ChannelSELayer(nb_filter[3], 2)
        self.cse3_3 = ChannelSELayer(nb_filter[2], 2)
        self.cse2_4 = ChannelSELayer(nb_filter[1], 2)
        self.cse1_5 = ChannelSELayer(nb_filter[0], 2)

        self.sse1_1 = SpatialSELayer(nb_filter[0])
        self.sse2_1 = SpatialSELayer(nb_filter[1])
        self.sse1_2 = SpatialSELayer(nb_filter[0])
        self.sse3_1 = SpatialSELayer(nb_filter[2])
        self.sse2_2 = SpatialSELayer(nb_filter[1])
        self.sse1_3 = SpatialSELayer(nb_filter[0])
        self.sse4_1 = SpatialSELayer(nb_filter[3])
        self.sse3_2 = SpatialSELayer(nb_filter[2])
        self.sse2_3 = SpatialSELayer(nb_filter[1])
        self.sse1_4 = SpatialSELayer(nb_filter[0])
        self.sse4_2 = SpatialSELayer(nb_filter[3])
        self.sse3_3 = SpatialSELayer(nb_filter[2])
        self.sse2_4 = SpatialSELayer(nb_filter[1])
        self.sse1_5 = SpatialSELayer(nb_filter[0])

        self.csse1_1 = ChannelSpatialSELayer(nb_filter[0], 2)
        self.csse2_1 = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse1_2 = ChannelSpatialSELayer(nb_filter[0], 2)
        self.csse3_1 = ChannelSpatialSELayer(nb_filter[2], 2)
        self.csse2_2 = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse1_3 = ChannelSpatialSELayer(nb_filter[0], 2)
        self.csse4_1 = ChannelSpatialSELayer(nb_filter[3], 2)
        self.csse3_2 = ChannelSpatialSELayer(nb_filter[2], 2)
        self.csse2_3 = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse1_4 = ChannelSpatialSELayer(nb_filter[0], 2)
        self.csse4_2 = ChannelSpatialSELayer(nb_filter[3], 2)
        self.csse3_3 = ChannelSpatialSELayer(nb_filter[2], 2)
        self.csse2_4 = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse1_5 = ChannelSpatialSELayer(nb_filter[0], 2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        if(self.useCSSE):
            conv1_1_ = self.csse1_1(self.conv1_1(x))
            pool1_ = self.max_pool(conv1_1_)

            conv2_1_ = self.csse2_1(self.conv2_1(pool1_))
            pool2_ = self.max_pool(conv2_1_)

            up1_2_ = self.up1_2(conv2_1_)
            conv1_2_ = torch.cat((up1_2_, conv1_1_), 1)
            conv1_2_ = self.csse1_2(self.conv1_2(conv1_2_))

            conv3_1_ = self.csse3_1(self.conv3_1(pool2_))
            pool3_ = self.max_pool(conv3_1_)

            up2_2_ = self.up2_2(conv3_1_)
            conv2_2_ = torch.cat((up2_2_, conv2_1_), 1)
            conv2_2_ = self.csse2_2(self.conv2_2(conv2_2_))

            up1_3_ = self.up1_3(conv2_2_)
            conv1_3_ = torch.cat((up1_3_, conv1_1_, conv1_2_), 1)
            conv1_3_ = self.csse1_3(self.conv1_3(conv1_3_))

            conv4_1_ = self.csse4_1(self.conv4_1(pool3_))
            pool4_ = self.max_pool(conv4_1_)

            up3_2_ = self.up3_2(conv4_1_)
            conv3_2_ = torch.cat((up3_2_, conv3_1_), 1)
            conv3_2_ = self.csse3_2(self.conv3_2(conv3_2_))

            up2_3_ = self.up2_3(conv3_2_)
            conv2_3_ = torch.cat((up2_3_, conv2_1_, conv2_2_), 1)
            conv2_3_ = self.csse2_3(self.conv2_3(conv2_3_))

            up1_4_ = self.up1_4(conv2_3_)
            conv1_4_ = torch.cat((up1_4_, conv1_1_, conv1_2_, conv1_3_), 1)
            conv1_4_ = self.csse1_4(self.conv1_4(conv1_4_))

            conv5_1_ = self.conv5_1(pool4_)

            up4_2_ = self.up4_2(conv5_1_)
            conv4_2_ = torch.cat((up4_2_, conv4_1_), 1)
            conv4_2_ = self.csse4_2(self.conv4_2(conv4_2_))

            up3_3_ = self.up3_3(conv4_2_)
            conv3_3_ = torch.cat((up3_3_, conv3_1_, conv3_2_), 1)
            conv3_3_ = self.csse3_3(self.conv3_3(conv3_3_))

            up2_4_ = self.up2_4(conv3_3_)
            conv2_4_ = torch.cat((up2_4_, conv2_1_, conv2_2_, conv2_3_), 1)
            conv2_4_ = self.csse2_4(self.conv2_4(conv2_4_))

            up1_5_ = self.up1_5(conv2_4_)
            conv1_5_ = torch.cat((up1_5_, conv1_1_, conv1_2_, conv1_3_, conv1_4_), 1)
            conv1_5_ = self.csse1_5(self.conv1_5(conv1_5_))

            output1 = self.conv0(conv1_2_)
            output2 = self.conv0(conv1_3_)
            output3 = self.conv0(conv1_4_)
            output4 = self.conv0(conv1_5_)

            conv1_ = torch.cat((output1, output2, output3, output4), 1)
            output = self.conv1(conv1_)

        elif(self.useCSE):
            conv1_1_ = self.cse1_1(self.conv1_1(x))
            pool1_ = self.max_pool(conv1_1_)

            conv2_1_ = self.cse2_1(self.conv2_1(pool1_))
            pool2_ = self.max_pool(conv2_1_)

            up1_2_ = self.up1_2(conv2_1_)
            conv1_2_ = torch.cat((up1_2_, conv1_1_), 1)
            conv1_2_ = self.cse1_2(self.conv1_2(conv1_2_))

            conv3_1_ = self.cse3_1(self.conv3_1(pool2_))
            pool3_ = self.max_pool(conv3_1_)

            up2_2_ = self.up2_2(conv3_1_)
            conv2_2_ = torch.cat((up2_2_, conv2_1_), 1)
            conv2_2_ = self.cse2_2(self.conv2_2(conv2_2_))

            up1_3_ = self.up1_3(conv2_2_)
            conv1_3_ = torch.cat((up1_3_, conv1_1_, conv1_2_), 1)
            conv1_3_ = self.cse1_3(self.conv1_3(conv1_3_))

            conv4_1_ = self.cse4_1(self.conv4_1(pool3_))
            pool4_ = self.max_pool(conv4_1_)

            up3_2_ = self.up3_2(conv4_1_)
            conv3_2_ = torch.cat((up3_2_, conv3_1_), 1)
            conv3_2_ = self.cse3_2(self.conv3_2(conv3_2_))

            up2_3_ = self.up2_3(conv3_2_)
            conv2_3_ = torch.cat((up2_3_, conv2_1_, conv2_2_), 1)
            conv2_3_ = self.cse2_3(self.conv2_3(conv2_3_))

            up1_4_ = self.up1_4(conv2_3_)
            conv1_4_ = torch.cat((up1_4_, conv1_1_, conv1_2_, conv1_3_), 1)
            conv1_4_ = self.cse1_4(self.conv1_4(conv1_4_))

            conv5_1_ = self.conv5_1(pool4_)

            up4_2_ = self.up4_2(conv5_1_)
            conv4_2_ = torch.cat((up4_2_, conv4_1_), 1)
            conv4_2_ = self.cse4_2(self.conv4_2(conv4_2_))

            up3_3_ = self.up3_3(conv4_2_)
            conv3_3_ = torch.cat((up3_3_, conv3_1_, conv3_2_), 1)
            conv3_3_ = self.cse3_3(self.conv3_3(conv3_3_))

            up2_4_ = self.up2_4(conv3_3_)
            conv2_4_ = torch.cat((up2_4_, conv2_1_, conv2_2_, conv2_3_), 1)
            conv2_4_ = self.cse2_4(self.conv2_4(conv2_4_))

            up1_5_ = self.up1_5(conv2_4_)
            conv1_5_ = torch.cat((up1_5_, conv1_1_, conv1_2_, conv1_3_, conv1_4_), 1)
            conv1_5_ = self.cse1_5(self.conv1_5(conv1_5_))

            output1 = self.conv0(conv1_2_)
            output2 = self.conv0(conv1_3_)
            output3 = self.conv0(conv1_4_)
            output4 = self.conv0(conv1_5_)

            conv1_ = torch.cat((output1, output2, output3, output4), 1)
            output = self.conv1(conv1_)

        elif(self.useSSE):
            conv1_1_ = self.sse1_1(self.conv1_1(x))
            pool1_ = self.max_pool(conv1_1_)

            conv2_1_ = self.sse2_1(self.conv2_1(pool1_))
            pool2_ = self.max_pool(conv2_1_)

            up1_2_ = self.up1_2(conv2_1_)
            conv1_2_ = torch.cat((up1_2_, conv1_1_), 1)
            conv1_2_ = self.sse1_2(self.conv1_2(conv1_2_))

            conv3_1_ = self.sse3_1(self.conv3_1(pool2_))
            pool3_ = self.max_pool(conv3_1_)

            up2_2_ = self.up2_2(conv3_1_)
            conv2_2_ = torch.cat((up2_2_, conv2_1_), 1)
            conv2_2_ = self.sse2_2(self.conv2_2(conv2_2_))

            up1_3_ = self.up1_3(conv2_2_)
            conv1_3_ = torch.cat((up1_3_, conv1_1_, conv1_2_), 1)
            conv1_3_ = self.sse1_3(self.conv1_3(conv1_3_))

            conv4_1_ = self.sse4_1(self.conv4_1(pool3_))
            pool4_ = self.max_pool(conv4_1_)

            up3_2_ = self.up3_2(conv4_1_)
            conv3_2_ = torch.cat((up3_2_, conv3_1_), 1)
            conv3_2_ = self.sse3_2(self.conv3_2(conv3_2_))

            up2_3_ = self.up2_3(conv3_2_)
            conv2_3_ = torch.cat((up2_3_, conv2_1_, conv2_2_), 1)
            conv2_3_ = self.sse2_3(self.conv2_3(conv2_3_))

            up1_4_ = self.up1_4(conv2_3_)
            conv1_4_ = torch.cat((up1_4_, conv1_1_, conv1_2_, conv1_3_), 1)
            conv1_4_ = self.sse1_4(self.conv1_4(conv1_4_))

            conv5_1_ = self.conv5_1(pool4_)

            up4_2_ = self.up4_2(conv5_1_)
            conv4_2_ = torch.cat((up4_2_, conv4_1_), 1)
            conv4_2_ = self.sse4_2(self.conv4_2(conv4_2_))

            up3_3_ = self.up3_3(conv4_2_)
            conv3_3_ = torch.cat((up3_3_, conv3_1_, conv3_2_), 1)
            conv3_3_ = self.sse3_3(self.conv3_3(conv3_3_))

            up2_4_ = self.up2_4(conv3_3_)
            conv2_4_ = torch.cat((up2_4_, conv2_1_, conv2_2_, conv2_3_), 1)
            conv2_4_ = self.sse2_4(self.conv2_4(conv2_4_))

            up1_5_ = self.up1_5(conv2_4_)
            conv1_5_ = torch.cat((up1_5_, conv1_1_, conv1_2_, conv1_3_, conv1_4_), 1)
            conv1_5_ = self.sse1_5(self.conv1_5(conv1_5_))

            output1 = self.conv0(conv1_2_)
            output2 = self.conv0(conv1_3_)
            output3 = self.conv0(conv1_4_)
            output4 = self.conv0(conv1_5_)

            conv1_ = torch.cat((output1, output2, output3, output4), 1)
            output = self.conv1(conv1_)

        else:
            conv1_1_ = self.conv1_1(x)
            pool1_ = self.max_pool(conv1_1_)

            conv2_1_ = self.conv2_1(pool1_)
            pool2_ = self.max_pool(conv2_1_)

            up1_2_ = self.up1_2(conv2_1_)
            conv1_2_ = torch.cat((up1_2_, conv1_1_), 1)
            conv1_2_ = self.conv1_2(conv1_2_)

            conv3_1_ = self.conv3_1(pool2_)
            pool3_ = self.max_pool(conv3_1_)

            up2_2_ = self.up2_2(conv3_1_)
            conv2_2_ = torch.cat((up2_2_, conv2_1_), 1)
            conv2_2_ = self.conv2_2(conv2_2_)

            up1_3_ = self.up1_3(conv2_2_)
            conv1_3_ = torch.cat((up1_3_, conv1_1_, conv1_2_), 1)
            conv1_3_ = self.conv1_3(conv1_3_)

            conv4_1_ = self.conv4_1(pool3_)
            pool4_ = self.max_pool(conv4_1_)

            up3_2_ = self.up3_2(conv4_1_)
            conv3_2_ = torch.cat((up3_2_, conv3_1_), 1)
            conv3_2_ = self.conv3_2(conv3_2_)

            up2_3_ = self.up2_3(conv3_2_)
            conv2_3_ = torch.cat((up2_3_, conv2_1_, conv2_2_), 1)
            conv2_3_ = self.conv2_3(conv2_3_)

            up1_4_ = self.up1_4(conv2_3_)
            conv1_4_ = torch.cat((up1_4_, conv1_1_, conv1_2_, conv1_3_), 1)
            conv1_4_ = self.conv1_4(conv1_4_)

            conv5_1_ = self.conv5_1(pool4_)

            up4_2_ = self.up4_2(conv5_1_)
            conv4_2_ = torch.cat((up4_2_, conv4_1_), 1)
            conv4_2_ = self.conv4_2(conv4_2_)

            up3_3_ = self.up3_3(conv4_2_)
            conv3_3_ = torch.cat((up3_3_, conv3_1_, conv3_2_), 1)
            conv3_3_ = self.conv3_3(conv3_3_)

            up2_4_ = self.up2_4(conv3_3_)
            conv2_4_ = torch.cat((up2_4_, conv2_1_, conv2_2_, conv2_3_), 1)
            conv2_4_ = self.conv2_4(conv2_4_)

            up1_5_ = self.up1_5(conv2_4_)
            conv1_5_ = torch.cat((up1_5_, conv1_1_, conv1_2_, conv1_3_, conv1_4_), 1)
            conv1_5_ = self.conv1_5(conv1_5_)

            output1 = self.conv0(conv1_2_)
            output2 = self.conv0(conv1_3_)
            output3 = self.conv0(conv1_4_)
            output4 = self.conv0(conv1_5_)

            conv1_ = torch.cat((output1, output2, output3, output4), 1)
            output = self.conv1(conv1_)
        
        return output

class Up_Only_U_Net_PP(nn.Module):
    def __init__(self, in_channels, out_channels, fc_dim=512, useBN=True):
        super(Up_Only_U_Net_PP, self).__init__()
        nb_filter = [fc_dim/16, fc_dim/8, fc_dim/4, fc_dim/2, fc_dim]
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1_1   = add_conv_stage(self.in_channels, nb_filter[0], useBN=useBN)
        self.conv2_1   = add_conv_stage(nb_filter[0], nb_filter[1], useBN=useBN)
        self.conv1_2   = add_conv_stage(nb_filter[0], nb_filter[0], useBN=useBN)
        self.conv3_1   = add_conv_stage(nb_filter[1], nb_filter[2], useBN=useBN)
        self.conv2_2   = add_conv_stage(nb_filter[0] + nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1_3   = add_conv_stage(nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)
        self.conv4_1   = add_conv_stage(nb_filter[2], nb_filter[3], useBN=useBN)
        self.conv3_2   = add_conv_stage(nb_filter[1] + nb_filter[2], nb_filter[2], useBN=useBN)
        self.conv2_3   = add_conv_stage(nb_filter[0] + nb_filter[1] + nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1_4   = add_conv_stage(nb_filter[0] + nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)
        self.conv5_1   = add_conv_stage(nb_filter[3], nb_filter[4], useBN=useBN)
        self.conv4_2   = add_conv_stage(nb_filter[2] + nb_filter[3] + nb_filter[3], nb_filter[3], useBN=useBN)
        self.conv3_3   = add_conv_stage(nb_filter[1] + nb_filter[2] + nb_filter[2] + nb_filter[2], nb_filter[2], useBN=useBN)
        self.conv2_4   = add_conv_stage(nb_filter[0] + nb_filter[1] + nb_filter[1] + nb_filter[1] + nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1_5   = add_conv_stage(nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)

        self.conv0 = nn.Conv2d(nb_filter[0], 1, 3, 1, 1)
        # self.conv1 = nn.Conv2d(4*self.in_channels, self.in_channels, 3, 1, 1)

        self.max_pool = nn.MaxPool2d(2)

        # self.up1_2 = upsample(nb_filter[1], nb_filter[0])
        # self.up2_2 = upsample(nb_filter[2], nb_filter[1])
        # self.up1_3 = upsample(nb_filter[1], nb_filter[0])
        # self.up3_2 = upsample(nb_filter[3], nb_filter[2])
        # self.up2_3 = upsample(nb_filter[2], nb_filter[1])
        # self.up1_4 = upsample(nb_filter[1], nb_filter[0])
        self.up4_2 = upsample(nb_filter[4], nb_filter[3])
        self.up3_3 = upsample(nb_filter[3], nb_filter[2])
        self.up2_4 = upsample(nb_filter[2], nb_filter[1])
        self.up1_5 = upsample(nb_filter[1], nb_filter[0])

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        
        conv1_1_ = self.conv1_1(x)
        pool1_1_ = self.max_pool(conv1_1_)

        conv2_1_ = self.conv2_1(pool1_1_)
        pool2_1_ = self.max_pool(conv2_1_)

        # up1_2_out = self.up1_2(conv2_1_out)
        # conv1_2_in = torch.cat((up1_2_out, conv1_1_out), 1)
        conv1_2_ = self.conv1_2(conv1_1_)
        pool1_2_ = self.max_pool(conv1_2_)

        conv3_1_ = self.conv3_1(pool2_1_)
        pool3_1_ = self.max_pool(conv3_1_)

        # up2_2_out = self.up2_2(conv3_1_out)
        conv2_2_ = torch.cat((pool1_2_, conv2_1_), 1)
        conv2_2_ = self.conv2_2(conv2_2_)
        pool2_2_ = self.max_pool(conv2_2_)

        # up1_3_out = self.up1_3(conv2_2_out)
        conv1_3_ = torch.cat((conv1_1_, conv1_2_), 1)
        conv1_3_ = self.conv1_3(conv1_3_)
        pool1_3_ = self.max_pool(conv1_3_)

        conv4_1_ = self.conv4_1(pool3_1_)
        pool4_1_ = self.max_pool(conv4_1_)

        # up3_2_out = self.up3_2(conv4_1_out)
        conv3_2_ = torch.cat((pool2_2_, conv3_1_), 1)
        conv3_2_ = self.conv3_2(conv3_2_)
        pool3_2_ = self.max_pool(conv3_2_)

        # up2_3_out = self.up2_3(conv3_2_out)
        conv2_3_ = torch.cat((pool1_3_, conv2_1_, conv2_2_), 1)
        conv2_3_ = self.conv2_3(conv2_3_)
        pool2_3_ = self.max_pool(conv2_3_)

        # up1_4_out = self.up1_4(conv2_3_out)
        conv1_4_ = torch.cat((conv1_1_, conv1_2_, conv1_3_), 1)
        conv1_4_ = self.conv1_4(conv1_4_)
        pool1_4_ = self.max_pool(conv1_4_)

        conv5_1_ = self.conv5_1(pool4_1_)

        up4_2_ = self.up4_2(conv5_1_)
        conv4_2_ = torch.cat((pool3_2_, up4_2_, conv4_1_), 1)
        conv4_2_ = self.conv4_2(conv4_2_)

        up3_3_ = self.up3_3(conv4_2_)
        conv3_3_ = torch.cat((pool2_3_, up3_3_, conv3_1_, conv3_2_), 1)
        conv3_3_ = self.conv3_3(conv3_3_)

        up2_4_ = self.up2_4(conv3_3_)
        conv2_4_ = torch.cat((pool1_4_, up2_4_, conv2_1_, conv2_2_, conv2_3_), 1)
        conv2_4_ = self.conv2_4(conv2_4_)

        up1_5_ = self.up1_5(conv2_4_)
        conv1_5_ = torch.cat((up1_5_, conv1_1_, conv1_2_, conv1_3_, conv1_4_), 1)
        conv1_5_ = self.conv1_5(conv1_5_)

        output1 = self.conv0(conv1_2_)
        output2 = self.conv0(conv1_3_)
        output3 = self.conv0(conv1_4_)
        output4 = self.conv0(conv1_5_)

        conv_out = torch.cat((output1, output2, output3, output4), 1)
        output = self.conv0(conv_out)
        
        return output