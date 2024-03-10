import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from Training_pkg.utils import *


activation_func = activation_function_table()


def resnet_block_lookup_table(blocktype):
    if blocktype == 'BasicBlock':
        return BasicBlock
    elif blocktype == 'Bottleneck':
        return Bottleneck
    else:
        print(' Wrong Key Word! BasicBlock or Bottleneck only! ')
        return None
    
def initial_network(width):

    if ResNet_setting:
        block = resnet_block_lookup_table(ResNet_Blocks)
        cnn_model = ResNet(nchannel=len(channel_names),block=block,blocks_num=ResNet_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)#cnn_model = Net(nchannel=nchannel)
    elif LateFusion_setting:
        block = resnet_block_lookup_table(ResNet_Blocks)
        cnn_model = LateFusion_ResNet(nchannel=len(LateFusion_initial_channels),nchannel_lf=len(LateFusion_latefusion_channels),block=block,blocks_num=ResNet_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)
    return cnn_model

class BasicBlock(nn.Module):  
    # F(X) and X has the same dimensions after two convolutional layers
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 1  # 残差映射F(X)的维度有没有发生变化，1表示没有变化，downsample=None
    # in_channel输入特征矩阵的深度(图像通道数，如输入层有RGB三个分量，使得输入特征矩阵的深度是3)，out_channel输出特征矩阵的深度(卷积核个数)，stride卷积步长，downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.tanh = activation_func
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity # out=F(X)+X
        out = self.tanh(out)

        return out
    
class Bottleneck(nn.Module):  
    # Three convolutional layers, F(x) and X have different dimensions.
    """
    注意: 原论文中, 在虚线残差结构的主分支上, 第一个1x1卷积层的步距是2, 第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1, 第二个3x3卷积层步距是2,
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.Tanh = activation_func
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out=F(X)+X
        out += identity
        out = self.Tanh(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 nchannel, # initial input channel
                 block,  # block types
                 blocks_num,  
                 num_classes=1,  
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.width_per_group = width_per_group
        self.actfunc = activation_func
        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,activation_func
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        #for m in self.modules(): 
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

   
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            x = self.fc(x)

        return x
        

class LateFusion_ResNet(nn.Module):

    def __init__(self,
                 nchannel, # initial input channel
                 nchannel_lf, # input channel for late fusion
                 block,  # block types
                 blocks_num,  
                 num_classes=1,  
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(LateFusion_ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_lf = 16
        self.groups = groups
        self.width_per_group = width_per_group
        self.actfunc = activation_func
        
        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,activation_func
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel_lf)
        ,activation_func
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[3], stride=1)

        self.layer1_lf = self._make_layer_lf(block, 32, blocks_num[0])
        self.layer2_lf = self._make_layer_lf(block, 64, blocks_num[1], stride=1)
        self.layer3_lf = self._make_layer_lf(block, 64, blocks_num[2], stride=1)

        self.fuse_layer = self._make_layer_fused(block, 512, blocks_num[2], stride=1)
        
                
        
        

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        #for m in self.modules(): 
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel_lf,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel_lf = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel_lf,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf+self.in_channel) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf+self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel_lf+self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x,x_lf):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_lf = self.layer0_lf(x_lf)
        x_lf = self.layer1_lf(x_lf)
        x_lf = self.layer2_lf(x_lf)
        x_lf = self.layer3_lf(x_lf)
        
        
        x = torch.cat((x,x_lf),1)
        x = self.fuse_layer(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            x = self.fc(x)

        return x
        
class Net(nn.Module):
    def __init__(self, nchannel):
        super(Net, self).__init__()

        self.conv = nn.Sequential(  # The first loop of ConvLay er, ReLU, Pooling
            nn.Conv2d(in_channels=nchannel,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64,momentum=0.1),
            nn.Tanh(),

            # self.conv2 = nn.Sequential(ResidualBlocks(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1),
            #                           nn.Tanh())

            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(128,momentum=0.1),
            nn.Tanh(),

            # self.conv4 = nn.Sequential(ResidualBlocks(in_channel=64,out_channel=64,kernel_size=3,stride=1,padding=1),
            #                           nn.Tanh())

            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(256,momentum=0.1),
            nn.Tanh(),
            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(512,momentum=0.1),
            nn.Tanh()
        )

        # self.ful1 = nn.Sequential(nn.Linear(256  * 5 * 5, 64), nn.BatchNorm1d(64))
        self.ful = nn.Sequential(nn.Linear(512 * 5 * 5, 64),  # , nn.BatchNorm1d(64), nn.Tanh())
                                 nn.Linear(64, 16),  # ,nn.BatchNorm1d(16), nn.Tanh())  # ,nn.Softmax())
                                 nn.Linear(16, 2),  # ,nn.BatchNorm1d(2), nn.Tanh())
                                 nn.Linear(2, 1))

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv(x)
        out = out.view(in_size, -1)
        output = self.ful(out)
        return output


