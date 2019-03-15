import cv2
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from entity import params, JointType
from models.CocoPoseNet import CocoPoseNet, compute_loss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from datetime import datetime
import time
from matplotlib import pyplot as plt

def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

class Openpose(object):
    def __init__(self, arch='posenet', weights_file=None, training = True):
        self.arch = arch
        if weights_file:
            self.model = params['archs'][arch]()
            self.model.load_state_dict(torch.load(weights_file))
        else:
            self.model = params['archs'][arch]()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if training:
            from coco_dataset import CocoDataset
            train_file = '/home/zzg/Datasets/UNICITY/data/splits/Argos_train.json'
            val_file = '/home/zzg/Datasets/UNICITY/data/splits/Argos_test.json'
            self.train_loader = DataLoader(CocoDataset(train_file, params['insize']), 
                                                                                params['batch_size'], 
                                                                                shuffle=True, 
                                                                                pin_memory=False,
                                                                                num_workers=params['num_workers'])
            self.val_loader = DataLoader(CocoDataset(val_file, params['insize'], mode = 'val'), 
                                                                                params['batch_size'], 
                                                                                shuffle=False, 
                                                                                pin_memory=False,
                                                                                num_workers=params['num_workers'])
            self.train_length = len(self.train_loader)
            self.val_length = len(self.val_loader)
            self.step = 0
            self.writer = SummaryWriter(params['log_path'])
            self.board_loss_every = self.train_length // params['board_loss_interval']
            self.evaluate_every = self.train_length // params['eval_interval']
            self.board_pred_image_every = self.train_length // params['board_pred_image_interval']
            self.save_every = self.train_length // params['save_interval']
            self.optimizer = Adam([{'params' : [*self.model.parameters()][:], 'lr' : params['lr']}])

    def board_scalars(self, key, loss, heatmap_log):
        self.writer.add_scalar('{}_loss'.format(key), loss, self.step)
        for stage, heatmap_loss in enumerate(heatmap_log):
            self.writer.add_scalar('{}_heatmap_loss_stage{}'.format(key, stage), heatmap_loss, self.step)

    def evaluate(self, num = 50):
        self.model.eval()
        count = 0
        running_loss = 0.
        running_heatmap_log = 0.
        with torch.no_grad():
            for heatmaps,imgs in iter(self.val_loader):
                imgs, heatmaps = imgs.to(self.device), heatmaps.to(self.device)
                heatmaps_ys = self.model(imgs)
                total_loss, heatmap_loss_log = compute_loss(heatmaps_ys, heatmaps)
                running_loss += total_loss.item()
                running_heatmap_log += heatmap_loss_log
                count += 1
                if count >= num:
                    break
        return running_loss / num, running_heatmap_log / num
    
    def save_state(self, val_loss, to_save_folder=False, model_only=False):
        if to_save_folder:
            save_path = params['work_space']+'save'
        else:
            save_path = params['work_space']+'model'
        time = get_time()
        torch.save(self.model.state_dict(), save_path+'/mode_{}_{}.pth'.format(val_loss,self.step))
        if not model_only:
            torch.save(self.optimizer.state_dict(), save_path+'/optimizer_{}_{}.pth'.format(val_loss,self.step))

    def load_state(self, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = params['work_space']/'save'
        else:
            save_path = params['work_space']/'model'          
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        print('load model_{}'.format(fixed_str))
        if not model_only:
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
            print('load optimizer_{}'.format(fixed_str))

    def lr_schedule(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10.
        print(self.optimizer)

    def train(self, resume = False):
        running_loss = 0.
        running_heatmap_log = 0.

        for epoch in range(60):
            for heatmaps, imgs in tqdm(iter(self.train_loader)):
#                heatmap = heatmaps.numpy()
#                heatshow1 = np.uint8(heatmap[0][0]*255)
#                heatshow2 = np.uint8(heatmap[0][1]*255)
#                heatshow3 = np.uint8(heatmap[0][2]*255)
#                heatshow4 = np.uint8(heatmap[0][3]*255)
#                cv2.imshow('h1t',heatshow1)
#                cv2.imshow('h2t',heatshow2)
#                cv2.imshow('h3t',heatshow3)
#                cv2.imshow('h4t',heatshow4)

                if self.step == 10000 or self.step == 20000:
                    self.lr_schedule()
                    
                imgs, heatmaps = imgs.to(self.device),heatmaps.to(self.device)
                self.optimizer.zero_grad()
                heatmaps_ys = self.model(imgs)

                with torch.no_grad():
                    htmp = heatmaps_ys[-1].clone()
                    heatmap = F.interpolate(htmp, (120, 160), mode='bilinear', align_corners=True).cpu().numpy()[0]
                    heatshow1 = np.uint8(heatmap[0]*255)
                    heatshow2 = np.uint8(heatmap[1]*255)
                    heatshow3 = np.uint8(heatmap[2]*255)
                    heatshow4 = np.uint8(heatmap[3]*255)
                    cv2.imshow('h1',heatshow1)
                    cv2.imshow('h2',heatshow2)
                    cv2.imshow('h3',heatshow3)
                    cv2.imshow('h4',heatshow4)
                    cv2.waitKey(10)
##                
                total_loss, heatmap_loss_log = compute_loss(heatmaps_ys, heatmaps)
                total_loss.backward()
                self.optimizer.step()
                
                running_loss += total_loss.item()
                running_heatmap_log += heatmap_loss_log

                if (self.step  % self.board_loss_every == 0) & (self.step != 0):
                    self.board_scalars('train', 
                                        running_loss / self.board_loss_every, 
                                        running_heatmap_log / self.board_loss_every)
                    running_loss = 0.
                    running_heatmap_log = 0.
                
                if (self.step  % self.evaluate_every == 0) & (self.step != 0):
                    val_loss, heatmap_loss_val_log = self.evaluate(num = params['eva_num'])
                    self.model.train()
                    self.board_scalars('val', val_loss, heatmap_loss_val_log)

                if (self.step  % self.save_every == 0) & (self.step != 0):
                    self.save_state(val_loss)
                
                self.step += 1
                if self.step > 300000:
                    break

    def pad_image(self, img, stride, pad_value):
        h, w, _ = img.shape

        pad = [0] * 2
        pad[0] = (stride - (h % stride)) % stride  # down
        pad[1] = (stride - (w % stride)) % stride  # right

        img_padded = np.zeros((h+pad[0], w+pad[1], 3), 'uint8') + pad_value
        img_padded[:h, :w, :] = img.copy()
        return img_padded, pad

    def compute_optimal_size(self, orig_img, img_size, stride=8):
        orig_img_h, orig_img_w, _ = orig_img.shape
        aspect = orig_img_h / orig_img_w
        if orig_img_h < orig_img_w:
            img_h = img_size
            img_w = np.round(img_size / aspect).astype(int)
            surplus = img_w % stride
            if surplus != 0:
                img_w += stride - surplus
        else:
            img_w = img_size
            img_h = np.round(img_size * aspect).astype(int)
            surplus = img_h % stride
            if surplus != 0:
                img_h += stride - surplus
        return (img_w, img_h)

    def compute_peaks_from_heatmaps(self, heatmaps):
        """all_peaks: shape = [N, 5], column = (jointtype, x, y, score, index)"""
        #heatmaps.shape : (4, h, w)
        #heatmaps[-1]  is background ,remove
        heatmaps = heatmaps[:-1]

        all_peaks = []
        peak_counter = 0
        for i , heatmap in enumerate(heatmaps):
            heatmap = gaussian_filter(heatmap, sigma=params['gaussian_sigma'])
            '''
            可以和下面的GPU codes对比一下，
            这里的gaussian_filter其实就是拿一个gaussian_kernel在输出的heatmaps上depth为1的平面卷积，
            因为网络拟合的heatmap也是在目标点上生成的gaussian heatmap,
            这样卷积比较合理地找到最贴近目标点的坐标
            '''
            map_left = np.zeros(heatmap.shape)
            map_right = np.zeros(heatmap.shape)
            map_top = np.zeros(heatmap.shape)
            map_bottom = np.zeros(heatmap.shape)
            '''
            我的理解，其实这里left和top, right和bottom搞反了，但是不影响最终结果
            '''
            map_left[1:, :] = heatmap[:-1, :]
            map_right[:-1, :] = heatmap[1:, :]
            map_top[:, 1:] = heatmap[:, :-1]
            map_bottom[:, :-1] = heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce((
                heatmap > params['heatmap_peak_thresh'],
                heatmap > map_left,
                heatmap > map_right,
                heatmap > map_top,
                heatmap > map_bottom,
            ))
            
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # [(x, y), (x, y)...]のpeak座標配列
            '''np.nonzero返回的坐标格式是[y,x],这里被改成了[x,y]'''
            peaks_with_score = [(i,) + peak_pos + (heatmap[peak_pos[1], peak_pos[0]],) for peak_pos in peaks]
            '''
            [(0, 387, 47, 0.050346997),
            (0, 388, 47, 0.050751492),
            (0, 389, 47, 0.051055912),
            .....]
            (关节点的index, x坐标, y坐标, heatmap value)
            '''
            peaks_id = range(peak_counter, peak_counter + len(peaks_with_score))
            peaks_with_score_and_id = [peaks_with_score[i] + (peaks_id[i], ) for i in range(len(peaks_id))]
            '''
            [(0, 387, 47, 0.050346997, 0),
            (0, 388, 47, 0.050751492, 1),
            (0, 389, 47, 0.051055912, 2),
            (0, 390, 47, 0.051255725, 3),
            ......]
            这一步还把序号带上了
            '''
            print(peaks_with_score_and_id)
            peak_counter += len(peaks_with_score_and_id)
            all_peaks.append(peaks_with_score_and_id)
        all_peaks = np.array([peak for peaks_each_category in all_peaks for peak in peaks_each_category])

        return all_peaks

    def detect(self, orig_img):
        orig_img = orig_img.copy()
        orig_img_h, orig_img_w = orig_img.shape

        x_data = torch.tensor(orig_img).to(self.device)
        x_data = x_data.unsqueeze(0).unsqueeze(0)
        x_data.requires_grad = False

        with torch.no_grad():

            heatmaps = self.model(x_data)
            heatmap = F.interpolate(heatmaps[-1], (orig_img_h, orig_img_w), mode='bilinear', align_corners=True).cpu().numpy()[0]
            print(heatmap.shape)
            heatshow1 = np.uint8(heatmap[0]*255)
            heatshow2 = np.uint8(heatmap[1]*255)
            heatshow3 = np.uint8(heatmap[2]*255)
            heatshow4 = np.uint8(heatmap[3]*255)
            cv2.imwrite('h1.png',heatshow1)
            cv2.imwrite('h2.png',heatshow2)
            cv2.imwrite('h3.png',heatshow3)
            cv2.imwrite('h4.png',heatshow4)

        all_peaks = self.compute_peaks_from_heatmaps(heatmap)

        if len(all_peaks) == 0:
            return all_peaks
        all_peaks[:, 1] *= orig_img_w / orig_img_w
        all_peaks[:, 2] *= orig_img_h / orig_img_h
        
        return all_peaks

def draw_person_pose(mat, poses):
    img = np.uint8(255*(mat+0.5))
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    joint_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    
    if(len(poses)==0):
        return img
    for pose in poses:
        cv2.circle(img, (int(pose[1]), int(pose[2])), 3, joint_colors[int(pose[0])], -1)
    return img

