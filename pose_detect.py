import cv2
import argparse
import numpy as np
from openpose import Openpose, draw_person_pose

def fix_values(data, min_value, max_value):
    data = data.copy()
    indx_1 = data > max_value
    indx_2 = data < min_value

    data[indx_1] = max_value
    data[indx_2] = min_value
    return data
    
def load_binary_file(path):
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
    
def process_depth_image(img, max_depth):
    img = fix_values(img, 0, max_depth)
    img = img.astype(np.float32)
    return img

def preprocess(img, max_depth):
    x_data = img.astype('f')
    x_data /= max_depth
    x_data -= 0.5
    return x_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img', '-i', help='image file path')
    parser.add_argument('--precise', '-p', action='store_true', help='do precise inference')
    parser.add_argument('--max_depth','-m', default=3500, type=int,help="max_depth")
    args = parser.parse_args()

    # load model
    openpose = Openpose(weights_file = args.weights, training = False)

    # read image
    mat = load_binary_file('/home/zzg/Datasets/UNICITY/data/recordings/N01N/063/ARGOS/depth/1513612398799_06471_00197.bin')
    mat = process_depth_image(mat, args.max_depth)
    mat = preprocess(mat,args.max_depth)

    # inference
    poses = openpose.detect(mat)
    print(poses)

    img = draw_person_pose(mat, poses)
    img = cv2.resize(img,(640,480))
#    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)



