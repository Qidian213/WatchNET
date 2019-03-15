import os
import sys
import cv2
import math
import random
import numpy as np
import torch
import json
from torch.utils.data import Dataset

from entity import JointType, params

class CocoDataset(Dataset):
    def __init__(self, json_file_name , insize, mode='train', n_samples=None):
        self.mode = mode
        self.insize = insize
        self.json_file = json.load(open(json_file_name))
        imagelists = []
        for key in self.json_file.keys():
            imagelists.append(key)
        self.imagelist = imagelists
        self.max_depth = 3500
        self.dataset_root = '/home/zzg/Datasets/UNICITY/'
        
    def __len__(self):
        return len(self.imagelist)

    def fix_values(self,data, min_value, max_value):
        data = data.copy()
        indx_1 = data > max_value
        indx_2 = data < min_value
        data[indx_1] = max_value
        data[indx_2] = min_value
        return data

    def load_binary_file(self,path):
        with open(path, "r") as fid:
            data = np.fromfile(fid, dtype=np.uint32)
            height = data[0] 
            width = data[1] 
            data_type = data[2]  
            num_channels = data[3]  
            
        with open(path, "rb") as fid:
            fid.read(4*4)
            if data_type == 2:
                data = np.fromfile(fid, dtype=np.uint16)
            elif data_type == 5:
                data = np.fromfile(fid, dtype=np.float32)
            img = np.reshape(data, (height, width)).astype(np.float)
        return img

    def process_depth_image(self,img, max_depth):
        img = self.fix_values(img, 0, max_depth)
        img = img.astype(np.float32)
        return img

    def parse_coco_annotation(self,ann_file):
        poses = np.zeros((0, len(JointType), 3), dtype=np.int32)
        for p in range(ann_file['num_people']):
            if not 'person_{}'.format(p) in ann_file.keys(): continue
            pose = np.zeros((1, len(JointType), 3), dtype=np.int32)
            
            person = ann_file['person_{}'.format(p)]
            for index ,lmk in enumerate(params['joint_list']):
                if not lmk in person.keys(): 
                    pose[0][index][0] = 0
                    pose[0][index][1] = 0
                    pose[0][index][2] = 0
                else:
                    pose[0][index][0] = person[lmk][0]
                    pose[0][index][1] = person[lmk][1]
                    pose[0][index][2] = 2
            poses = np.vstack((poses, pose))
        return poses

    def generate_gaussian_heatmap(self, shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap
        
    def generate_heatmaps(self,img, poses, heatmap_sigma):
        heatmaps = np.zeros((0,) + img.shape)
        sum_heatmap = np.zeros(img.shape)
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape)
            for pose in poses:
                if pose[joint_index,2]>0:
                    jointmap = self.generate_gaussian_heatmap(img.shape, pose[joint_index][:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        '''
        We take the maximum of the confidence maps insteaof the average so that thprecision of close by peaks remains distinct, 
        as illus- trated in the right figure. At test time, we predict confidence maps (as shown in the first row of Fig. 4), 
        and obtain body part candidates by performing non-maximum suppression.
        At test time, we predict confidence maps (as shown in the first row of Fig. 4), 
        and obtain body part candidates by performing non-maximum suppression.
        '''
        return heatmaps.astype('f')

    def generate_labels(self, img , poses):
        heatmaps = self.generate_heatmaps(img, poses, params['heatmap_sigma'])
        return heatmaps

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= self.max_depth
        x_data -= 0.5
        return x_data

    def __getitem__(self, i):
        sample_key = self.imagelist[i]
        ann_path = self.json_file[sample_key]['ann_path']
        img_path = self.json_file[sample_key]['img_path']
        ann_path = self.dataset_root+ann_path
        img_path = self.dataset_root+img_path
        
        mat = self.load_binary_file(img_path)
        mat = self.process_depth_image(mat,self.max_depth)
#        img = np.uint8(255*mat/self.max_depth)

        ann_file = json.load(open(ann_path))
        poses = self.parse_coco_annotation(ann_file)
        heatmaps = self.generate_labels(mat, poses)
        mat = self.preprocess(mat)
        mat = torch.tensor(mat)
        mat = mat.unsqueeze(0)
        heatmaps = torch.tensor(heatmaps)
        return heatmaps,mat




