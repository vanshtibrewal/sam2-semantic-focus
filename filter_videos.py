import os
import cv2
from tqdm import tqdm
import numpy as np

video_data_path = "/home/vansh/Desktop/labelled_data"

for vid_name in os.listdir(video_data_path):
    print(vid_name)
    vid_path = os.path.join(video_data_path, vid_name)
    video_base_name = os.path.splitext(vid_name)[0]
    output_dir = os.path.join(video_data_path, video_base_name)
    top_dir = os.path.join(output_dir, 'top')
    bottom_dir = os.path.join(output_dir, 'bottom')
    
    for img_name in tqdm(sorted(os.listdir(top_dir), key=lambda x: int(x.split('.')[0]))):
        img_path = os.path.join(top_dir, img_name)
        img = cv2.imread(img_path)
        blue_channel, green_channel, red_channel = cv2.split(img)
        max_val_blue, max_val_green, max_val_red = np.max(blue_channel), np.max(green_channel), np.max(red_channel)
        min_val_blue, min_val_green, min_val_red = np.min(blue_channel), np.min(green_channel), np.min(red_channel)
        black = max_val_blue < 200 and max_val_green < 200 and max_val_red < 200
        white = min_val_blue > 50 and min_val_green > 50 and min_val_red > 50
        if black or white:
            os.remove(img_path)

    for img_name in tqdm(sorted(os.listdir(bottom_dir), key=lambda x: int(x.split('.')[0]))):
        img_path = os.path.join(bottom_dir, img_name)
        img = cv2.imread(img_path)
        blue_channel, green_channel, red_channel = cv2.split(img)
        max_val_blue, max_val_green, max_val_red = np.max(blue_channel), np.max(green_channel), np.max(red_channel)
        min_val_blue, min_val_green, min_val_red = np.min(blue_channel), np.min(green_channel), np.min(red_channel)
        black = max_val_blue < 200 and max_val_green < 200 and max_val_red < 200
        white = min_val_blue > 50 and min_val_green > 50 and min_val_red > 50
        if black or white:
            os.remove(img_path)

 