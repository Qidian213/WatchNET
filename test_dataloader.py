import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from coco_dataset import CocoDataset

train_loader = DataLoader(CocoDataset('/home/zzg/Datasets/UNICITY/data/splits/Argos_train.json', 368),6, shuffle=True, pin_memory=False,num_workers=2)

for i, (heatmaps,mat) in tqdm(enumerate(train_loader), total=len(train_loader)):
    print(mat.size())
    mat = mat.numpy()
    heatmaps = heatmaps.numpy()
    heatshow1 = np.uint8(heatmaps[0][0]*255)
    heatshow2 = np.uint8(heatmaps[0][1]*255)
    heatshow3 = np.uint8(heatmaps[0][2]*255)
    heatshow4 = np.uint8(heatmaps[0][3]*255)
    cv2.imshow('h1',heatshow1)
    cv2.imshow('h2',heatshow2)
    cv2.imshow('h3',heatshow3)
    cv2.imshow('h4',heatshow4)
    
    img = np.uint8(255*(mat[0]+0.5))

    cv2.waitKey(1000)


