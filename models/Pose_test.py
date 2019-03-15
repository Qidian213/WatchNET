import torch
import numpy as np
from CocoPoseNet import CocoPoseNet
#from visnet import make_dot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CocoPoseNet()
model = model.to(device)

x = np.ones([368,368])
x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
x = x.to(device)

print(x.size())
heatmaps = model(x)

for heatmap in heatmaps:
    print(heatmap.shape)
    
#g = make_dot(heatmaps[2])
#g.view()
