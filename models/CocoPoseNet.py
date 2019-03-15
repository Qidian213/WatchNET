import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d ,init
import torch.nn.functional as F
import numpy as np

def compute_loss(heatmaps_ys, heatmaps_t):
    heatmap_loss_log = []
    total_loss = 0

    # compute loss on each stage
    for heatmaps_y in heatmaps_ys:
        stage_heatmaps_t = heatmaps_t.clone()
        
        if heatmaps_y.shape != stage_heatmaps_t.shape:
            with torch.no_grad():
                stage_heatmaps_t = F.interpolate(stage_heatmaps_t, heatmaps_y.shape[2:], mode='bilinear', align_corners=True)

        heatmaps_loss = mean_square_error(heatmaps_y, stage_heatmaps_t)

        total_loss += heatmaps_loss
        
        heatmap_loss_log.append(heatmaps_loss.item())

    return total_loss, np.array(heatmap_loss_log)

def mean_square_error(pred, target):
    assert pred.shape == target.shape, 'x and y should in same shape'
    return torch.sum((pred - target) ** 2) / target.nelement()

class CocoPoseNet(Module):
    insize = 368
    def __init__(self, path = None):
        super(CocoPoseNet, self).__init__()
        self.base = Base_model()
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)
        if path:
            self.base.vgg_base.load_state_dict(torch.load(path))
        
    def forward(self, x):
        heatmaps = []
        feature_map = self.base(x)
        p1 = self.stage_1(feature_map)
        heatmaps.append(p1)

        p2 = self.stage_2(torch.cat((p1, feature_map), dim = 1))
        heatmaps.append(p2)
        
        p3 = self.stage_3(torch.cat((p2, feature_map), dim = 1))
        heatmaps.append(p3)
        
        return heatmaps
        
class VGG_Base(Module):
    def __init__(self):
        super(VGG_Base, self).__init__()
        self.conv1 = Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 1, stride = 1, padding = 0)
        self.conv5 = Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = Conv2d(in_channels = 128, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv7 = Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv3(x))
        c4_x = self.relu(self.conv4(x))
        c5_x = self.max_pooling_2d(x)
        c5_x = self.relu(self.conv5(c5_x))
        c5_x = F.interpolate(c5_x,scale_factor=2,mode ='bilinear',align_corners=True)
        c6_x = torch.cat((c4_x,c5_x),dim = 1)
        c7_x = self.relu(self.conv6(c6_x))
        feature = self.relu(self.conv7(c7_x))
        return feature

class Base_model(Module):
    def __init__(self):
        super(Base_model, self).__init__()
        self.vgg_base = VGG_Base()
        self.conv1_CPM = Conv2d(in_channels=64, out_channels=64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2_CPM = Conv2d(in_channels=64, out_channels=64,  kernel_size = 3, stride = 1, padding = 1)
        self.relu = ReLU()
    def forward(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv1_CPM(x))
        x = self.relu(self.conv2_CPM(x))
        return x

class Stage_1(Module):
    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv1 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = Conv2d(in_channels = 64, out_channels = 32, kernel_size = 1, stride = 1, padding = 0)
        self.conv4 = Conv2d(in_channels = 32, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
        self.relu = ReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1(x))
        h1 = self.relu(self.conv2(h1))
        h1 = self.relu(self.conv3(h1))
        h1 = self.relu(self.conv4(h1))
        return h1

class Stage_x(Module):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1 = Conv2d(in_channels = 68, out_channels = 64, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = Conv2d(in_channels = 64, out_channels = 32, kernel_size = 1, stride = 1, padding = 0)
        self.conv4 = Conv2d(in_channels = 32, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
        self.relu = ReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1(x))
        h1 = self.relu(self.conv2(h1))
        h1 = self.relu(self.conv3(h1))
        h1 = self.relu(self.conv4(h1))
        return h1
