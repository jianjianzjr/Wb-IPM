import random
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class InitialCONVNet(nn.Module):
    def __init__(self):
        super(InitialCONVNet, self).__init__()
        # create network model
        self.attn_2_1 = None
        self.attn_1_1 = None
        self.attn_3_1 = None
        self.attn_4_1 = None
        self.block_1_1 = None
        self.block_2_1 = None
        self.block_3_1 = None
        self.block_4_1 = None
        self.block_5 = None
        self.block_4_2 = None
        self.block_3_2 = None
        self.block_2_2 = None
        self.block_1_2 = None
        self.thres = None
        # self.threshold = nn.Parameter(torch.tensor([initial_threshold]))
        self.create_model()

    def forward(self, input_data, excitation):
        # Apply the mask based on the threshold
        # mask = excitation > (self.threshold * excitation.max())
        # masked_input_data = input_data * mask.float()
        # masked_input_data = self.thres(input_data, excitation, batchOn=True, ReluOn=True)

        # U-shaped network with multi-head attention model
        block_1_1_output = self.block_1_1(input_data)
        block_1_1_attn= self.attn_1_1(block_1_1_output)
        block_2_1_output = self.block_2_1(block_1_1_output)
        block_2_1_attn = self.attn_2_1(block_2_1_output)
        block_3_1_output = self.block_3_1(block_2_1_output)
        block_3_1_attn = self.attn_3_1(block_3_1_output)
        block_4_1_output = self.block_4_1(block_3_1_output)
        block_4_1_attn = self.attn_4_1(block_4_1_output)
        block_5_output = self.block_5(block_4_1_output)
        # block_4_1_attn = self.attn_4_1(block_4_1_output)
        # block_5_output = self.block_5(block_4_1_output)
        # result = self.block_4_2(torch.cat((block_4_1_output, block_5_output), dim=1))
        # result = self.block_4_2block_4_1_output()
        result = self.block_4_2(torch.cat((block_4_1_attn, block_5_output), dim=1))
        result = self.block_3_2(torch.cat((block_3_1_attn, result), dim=1))
        result = self.block_2_2(torch.cat((block_2_1_attn, result), dim=1))
        result = self.block_1_2(torch.cat((block_1_1_attn, result), dim=1))
        # result = result + input_data
        return result

    def create_model(self):
        kernel_size = 3
        padding = kernel_size // 2

        self.thres = LearnableThresholdLayer(in_channel=100)
        # block_1_1
        block_1_1 = []
        block_1_1.extend(self.add_block_conv(in_channels=100, out_channels=128, kernel_size=2, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_1_1 = nn.Sequential(*block_1_1)

        self.attn_1_1 = SelfAttention(in_channel=128, n_head=4)

        # block_2_1
        block_2_1 = [nn.MaxPool2d(kernel_size=2)]
        block_2_1.extend(self.add_block_conv(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_1.extend(self.add_block_conv(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_2_1 = nn.Sequential(*block_2_1)

        self.attn_2_1 = SelfAttention(in_channel=256, n_head=4)

        # block_3_1
        block_3_1 = [nn.MaxPool2d(kernel_size=2)]
        block_3_1.extend(self.add_block_conv(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_1.extend(self.add_block_conv(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_3_1 = nn.Sequential(*block_3_1)

        self.attn_3_1 = SelfAttention(in_channel=512, n_head=4)

        # block_4_1
        block_4_1 = [nn.MaxPool2d(kernel_size=2)]
        block_4_1.extend(self.add_block_conv(in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_1.extend(self.add_block_conv(in_channels=1024, out_channels=1024, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_4_1 = nn.Sequential(*block_4_1)

        self.attn_4_1 = SelfAttention(in_channel=1024, n_head=4)

        # nlock 5
        block_5 = []
        block_5.extend(self.add_block_conv(in_channels=1024, out_channels=1024, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))

        self.block_5 = nn.Sequential(*block_5)

        # block_5
        # block_5 = [nn.MaxPool2d(kernel_size=2)]
        # block_5.extend(self.add_block_conv(in_channels=1024, out_channels=2048, kernel_size=kernel_size, stride=1,
        #                                    padding=padding, batchOn=True, ReluOn=True))
        # block_5.extend(self.add_block_conv(in_channels=2048, out_channels=2048, kernel_size=kernel_size, stride=1,
        #                                    padding=padding, batchOn=True, ReluOn=True))
        # block_5.extend(self.add_block_conv_transpose(in_channels=2048, out_channels=1024, kernel_size=kernel_size, stride=2,
        #                                              padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        # block_5.extend(self.add_block_conv(in_channels=1024, out_channels=1024, kernel_size=2, stride=1,
        #                                    padding=padding, batchOn=True, ReluOn=True))
        # self.block_5 = nn.Sequential(*block_5)

        # block_4_2
        block_4_2 = []
        block_4_2.extend(self.add_block_conv(in_channels=2048, out_channels=1024, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(self.add_block_conv(in_channels=1024, out_channels=1024, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(
            self.add_block_conv_transpose(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_4_2 = nn.Sequential(*block_4_2)



        # block_3_2
        block_3_2 = []
        block_3_2.extend(self.add_block_conv(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(self.add_block_conv(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(
            self.add_block_conv_transpose(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_3_2 = nn.Sequential(*block_3_2)



        # block_2_2
        block_2_2 = []
        block_2_2.extend(self.add_block_conv(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(self.add_block_conv(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(
            self.add_block_conv_transpose(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_2_2 = nn.Sequential(*block_2_2)



        # block_1_2
        block_1_2 = []
        block_1_2.extend(self.add_block_conv(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=64, out_channels=15, kernel_size=2, stride=1,
                                             padding=0, batchOn=True, ReluOn=False, SigmoidOn=True))
        self.block_1_2 = nn.Sequential(*block_1_2)



    @staticmethod
    def add_block_conv(in_channels, out_channels, kernel_size, stride, padding, batchOn, ReluOn, SigmoidOn=False):
        seq = []

        # conv layer
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding)
        nn.init.normal_(conv.weight, 0, 0.01)
        nn.init.constant_(conv.bias, 0)
        seq.append(conv)
        # batch norm layer
        if batchOn:
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            nn.init.constant_(batch_norm.weight, 1)
            nn.init.constant_(batch_norm.bias, 0)
            seq.append(batch_norm)
        # relu layer
        if ReluOn:
            seq.append(nn.ReLU())
        # sigmoid layer
        if SigmoidOn:
            seq.append(nn.Sigmoid())

        return seq

    @staticmethod
    def add_block_conv_transpose(in_channels, out_channels, kernel_size, stride, padding, output_padding, batchOn, ReluOn):
        seq = []

        convt = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, output_padding=output_padding)
        nn.init.normal_(convt.weight, 0, 0.01)
        nn.init.constant_(convt.bias, 0)
        seq.append(convt)

        if batchOn:
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            nn.init.constant_(batch_norm.weight, 1)
            nn.init.constant_(batch_norm.bias, 0)
            seq.append(batch_norm)

        if ReluOn:
            seq.append(nn.ReLU())

        return seq

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        self.n_head = n_head

        # self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

        self.norm = nn.BatchNorm2d(num_features=in_channel)
        nn.init.constant_( self.norm.weight, 1)
        nn.init.constant_( self.norm.bias, 0)

        self.relu = nn.ReLU()

    def forward(self, input_data, batchOn=True, ReluOn=True):
        batch, channel, height, width = input_data.shape
        n_head = self.n_head
        head_dim = channel // n_head

        qkv = self.qkv(input_data).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        if batchOn:
            out = self.norm(out)
        if ReluOn:
            out = self.relu(out)


        return out + input_data

class LearnableThresholdLayer(nn.Module):
    def __init__(self, in_channel=100, initial_thres=0.001, lower_bound=1e-6):
        super().__init__()
        self.threshold = nn.Parameter(torch.Tensor([initial_thres]), requires_grad=True)
        # self.threshold = torch.Tensor([0.05])
        self.lower_bound = lower_bound
        # torch.cuda.manual_seed_all(123)
        ##### mask via a bounded area ########
        # center = 25 + (30 - 25) * torch.rand(in_channel, 2)
        # # Last two columns: values in the range [10, 30]
        # radius = 10 + (15 - 10) * torch.rand(in_channel, 2)
        # center = 55.0 / 2 * torch.ones(in_channel, 2)
        # radius = 5.0 * torch.ones(in_channel, 2)
        # combined = torch.cat((center, radius), dim=1)
        # self.para = nn.Parameter(combined, requires_grad=True)

        # self.threshold =0.05

        self.norm = nn.BatchNorm2d(num_features=in_channel)
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

        self.relu = nn.ReLU()

    def forward(self, input_data, excitation, batchOn=True, ReluOn=True):

        # sigmoid_threshold = F.sigmoid(self.threshold)


        #### approximate via tanh ##########

        self.threshold.data = torch.clamp(self.threshold.data, min=self.lower_bound)
        thres_value = self.threshold * excitation.max()

        # method 1 tanh
        ex_thres = excitation - thres_value
        filtered_data = ex_thres[ex_thres != 0]
        epi = 8e-2
        eplison = epi / (filtered_data.abs().min() ** 1)
        mask = (torch.tanh(ex_thres * eplison) + 1) / 2

        # method 2
        # ex_thres = self.relu(excitation - thres_value)
        # filtered_data = ex_thres[ex_thres != 0]
        # epi = 1
        # eplison = epi*np.min(np.abs(filtered_data))
        # mask = ex_thres/(ex_thres+eplison)

        # mask = self.relu(excitation - thres_value)
        # mask = torch.where(excitation >= thres_value, excitation, torch.tensor(0.00))
        # mask = mask/(mask+eplison)

        # out = input_data * mask

        ####  ##### mask via a bounded area ########

        # Get the shape of the input data

        # array_shape = excitation.shape
        # mask = torch.zeros(array_shape)


        # for i in range(array_shape[1]):
        #     # Create grid of coordinates
        #     Y, X = torch.meshgrid(torch.arange(array_shape[2], device=input_data.device),
        #                           torch.arange(array_shape[3], device=input_data.device), indexing='ij')
        #
        #     # Calculate the distance from the center
        #     distance_from_center = ((X - self.para[i, 0]) ** 2) / (self.para[i, 2] ** 2) + ((Y - self.para[i, 1]) ** 2) / (
        #                 self.para[i, 3] ** 2)
        #
        #     mask_slice = (torch.tanh((1-distance_from_center)/epsilon) + 1)/2
        #     mask[0, i, :, :] = mask_slice

        mask = mask.to(input_data.device)
        out = mask*input_data


        if batchOn:
            out = self.norm(out)
        if ReluOn:
            out = self.relu(out)
        return out
