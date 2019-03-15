from enum import IntEnum

from models.CocoPoseNet import CocoPoseNet

from pathlib import Path

class JointType(IntEnum):
    Head = 0
    """ 头 """
    LeftShoulder = 1
    """ 左肩 """
    RightShoulder = 2
    """ 右肩 """

params = {
    'coco_dir': '/home/zzg/Datasets/coco2017',
    'archs': {
        'posenet': CocoPoseNet,
    },
    'pretrained_path' : 'models/pretrained_vgg_base.pth',
    # training params
    'min_keypoints': 5,
    'min_area': 32 * 32,
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,
    'batch_size': 6,
    'lr': 1e-4,
    'num_workers': 2,
    'eva_num': 2,
    'board_loss_interval': 500,
    'eval_interval': 100,
    'board_pred_image_interval': 2,
    'save_interval': 8,
    'log_path': 'work_space/log',
    'work_space': 'work_space/',
    
    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,

    # inference params
    'inference_img_size': 368,
    'inference_scales': [0.5, 1, 1.5, 2],
    # 'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.1,
    'inner_product_thresh': 0.05,
    'limb_length_ratio': 1.0,
    'length_penalty_value': 1,
    'n_subset_limbs_thresh': 3,
    'subset_score_thresh': 0.2,
    'limbs_point': [
        [JointType.LeftShoulder, JointType.Head],
        [JointType.RightShoulder, JointType.Head],
        [JointType.LeftShoulder, JointType.RightShoulder],
    ],
    'coco_joint_indices': [
        JointType.Head,
        JointType.LeftShoulder,
        JointType.RightShoulder,
    ],
    'joint_list':[
     'head', 
     'l_shoulder', 
     'r_shoulder',
    ]
}
